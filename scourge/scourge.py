import uuid
import json

from functools import partial
from multiprocessing import cpu_count
from concurrent import futures

import click
import sh
import toolz


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


@cli.command(help='Generate a Makefile for building packages')
@click.argument('packages', nargs=-1)
def gen(packages):
    command = sh.conda.create.bake(
        name=uuid.uuid4().hex,
        dry_run=True,
        json=True,
        channel='conda-forge'
    )

    get_package_data = toolz.compose(json.loads, str, command)

    graph = {package: set() for package in packages}

    with futures.ThreadPoolExecutor(
        max_workers=min(cpu_count(), len(packages))
    ) as e:
        results = futures.as_completed(
            map(partial(e.submit, get_package_data), packages),
        )
        for package, package_data in zip(
            packages,
            (future.result() for future in results)
        ):
            action, *_ = package_data['actions']
            link = action['LINK']
            for dep in link:
                dep_name = dep['name']
                if dep_name in packages and dep_name != package:
                    graph[package].add(dep_name)
    template = """
{}: {}
\t[ ! -d "$@-feedstock" ] && git clone git://github.com/conda-forge/$@-feedstock
\t$@-feedstock/ci_support/run_docker_build.sh
"""
    lines = ['all: {}\n'.format(' '.join(graph.keys()))]
    lines += [
        template.format(node, ' '.join(edges))
        for node, edges in graph.items()
    ]
    click.echo('\n'.join(lines))
