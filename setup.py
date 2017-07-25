from setuptools import setup, find_packages

import versioneer

with open('README.md', 'rt') as f:
    long_description = f.read()

with open('requirements.txt', 'rt') as f:
    install_requires = list(map(str.strip, f))

setup(
    name='scourge',
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    description="Build arbitrary conda dependency graphs from source",
    long_description=long_description,
    license='Apache',
    maintainer="Phillip Cloud",
)
