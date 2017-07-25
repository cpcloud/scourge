# scourge

Build arbitrary conda dependency graphs from source


## Example

Here's an example of how to use `scourge`.


### Initialization

Right now all this does is pull down the `condaforge/linux-anvil` docker image.

```sh
$ scourge init
Using default tag: latest
latest: Pulling from condaforge/linux-anvil
Digest: sha256:dd53ef1792f777fa8b2abf30221bf9beb6cb6a81d158d498f0219c4d475696cd
Status: Image is up to date for condaforge/linux-anvil:latest
```

## `Makefile` generation

This writes to standard out a `Makefile` which will build your packages in
topological order. `scourge` just generates the package name and its dependency
list and `make` does the sorting.

```sh
$ scourge gen parquet-cpp arrow-cpp pyarrow
all: parquet-cpp arrow-cpp pyarrow


parquet-cpp: arrow-cpp
        [ ! -d "$@-feedstock" ] && git clone git://github.com/conda-forge/$@-feedstock
        $@-feedstock/ci_support/run_docker_build.sh


arrow-cpp: parquet-cpp
        [ ! -d "$@-feedstock" ] && git clone git://github.com/conda-forge/$@-feedstock
        $@-feedstock/ci_support/run_docker_build.sh


pyarrow: arrow-cpp parquet-cpp
        [ ! -d "$@-feedstock" ] && git clone git://github.com/conda-forge/$@-feedstock
        $@-feedstock/ci_support/run_docker_build.sh
```
