import os
import pathlib
import io

from multiprocessing import cpu_count
from concurrent import futures

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import click
import sh


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command(help='Pull in the conda forge docker image')
@click.argument(
    'docker_image', required=False, default='condaforge/linux-anvil',
)
def init(docker_image):
    # pull passed in docker image
    click.echo(sh.docker.pull(docker_image, _fg=True))


@cli.command(help='Clone package feedstocks to build')
@click.argument('packages', nargs=-1)
def clone(packages):
    with futures.ThreadPoolExecutor(
        max_workers=min(cpu_count(), len(packages))
    ) as executor:
        for future in futures.as_completed(
            executor.submit(
                sh.git.clone,
                'https://github.com/conda-forge/{}'.format(feedstock),
            ) for feedstock in map('{}-feedstock'.format, packages)
            if not os.path.exists(feedstock)
        ):
            try:
                future.result()
            except Exception as e:
                raise click.ClickException(str(e))


def get_requirements(packages):
    with futures.ThreadPoolExecutor() as e:
        for future in futures.as_completed(
            e.submit(sh.conda.render, str(package / 'recipe' / 'meta.yaml'))
            for package_name, package in packages.items()
        ):
            try:
                result = future.result()
            except Exception as e:
                raise click.ClickException(str(e))
            else:
                meta = yaml.load(io.StringIO(str(result)), Loader=Loader)
                yield meta['package']['name'], meta['requirements']


@cli.command(help='Generate a Makefile for building packages')
@click.argument(
    'output',
    required=False,
    default=click.get_text_stream('stdout'),
    type=click.File('wt'),
)
@click.option(
    '-d', '--feedstock-directory',
    default='.',
    type=click.Path(exists=True),
    help='Directory of feedstocks',
)
@click.option(
    '-a', '--artifact-directory',
    default=os.path.join('.', 'artifacts'),
    type=click.Path(),
    help='Where to place tarballs upon building successfully',
)
def gen(output, feedstock_directory, artifact_directory):
    packages = {
        p.stem[:-len('-feedstock')]: p
        for p in pathlib.Path(
            feedstock_directory
        ).glob('*-feedstock') if p.is_dir()
    }

    graph = {package_name: set() for package_name in packages.keys()}

    for package_name, requirements in get_requirements(packages):
        for dep in frozenset(requirements['build'] + requirements['run']):
            raw_dep, *_ = dep.split(' ', 1)
            if raw_dep in packages:
                graph.setdefault(package_name, set()).add(raw_dep)

        os.makedirs(
            os.path.join(artifact_directory, package_name),
            exist_ok=True,
        )

    target_template = '{}/{{}}/BUILT'.format(artifact_directory)

    template = """
{}/{{package_name}}/BUILT: {{deps}}
\t{{stem}}/ci_support/run_docker_build.sh
\tcp {{stem}}/build_artefacts/linux-64/{{package_name}}*.tar.bz2 $(dir $@)
\ttouch $@""".format(artifact_directory)
    lines = [
        '.PHONY: all clean realclean\n',
        'all: {}\n'.format(
            ' '.join(map(target_template.format, graph.keys()))
        ),
        'clean:\n\trm -f {}/*/BUILT\n'.format(artifact_directory),
        'realclean:\n\trm -rf {}/*'.format(artifact_directory)
    ]
    lines += [
        template.format(
            package_name=node,
            deps=' '.join(map(target_template.format, edges)),
            stem='{}-feedstock'.format(node),
        )
        for node, edges in graph.items()
    ]
    output.write('\n'.join(lines) + '\n')
