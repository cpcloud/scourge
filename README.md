# scourge

Build arbitrary conda dependency graphs from source


## Example

Here's an example of how to use `scourge`.


### 1. Initialization: `scourge init`

Right now all this does is pull down the `condaforge/linux-anvil` docker image.

```sh
$ scourge init
Using default tag: latest
latest: Pulling from condaforge/linux-anvil
Digest: sha256:dd53ef1792f777fa8b2abf30221bf9beb6cb6a81d158d498f0219c4d475696cd
Status: Image is up to date for condaforge/linux-anvil:latest
```

### 2. Feedstock clone: `scourge clone <packages>`

This step clones each package's feedstock from the `conda-forge` GitHub
repository into the current working directory.

```sh
$ scourge clone parquet-cpp arrow-cpp pyarrow
```

There's no output if cloning is successful.

### 3. `Makefile` generation: `scourge gen`

This writes to standard out (or a file name) a `Makefile` which will
build your packages in topological order.

`scourge` generates the package name and its dependency list and `make` does
the sorting.

```sh
$ scourge gen -
.PHONY: all clean realclean

all: ./artifacts/pyarrow/BUILT ./artifacts/parquet-cpp/BUILT ./artifacts/arrow-cpp/BUILT

clean:
        rm -f ./artifacts/*/BUILT

realclean:
        rm -rf ./artifacts/*


./artifacts/pyarrow/BUILT: ./artifacts/parquet-cpp/BUILT ./artifacts/arrow-cpp/BUILT
        pyarrow-feedstock/ci_support/run_docker_build.sh
        cp pyarrow-feedstock/build_artefacts/linux-64/pyarrow*.tar.bz2 $(dir $@)
        touch $@

./artifacts/parquet-cpp/BUILT: ./artifacts/arrow-cpp/BUILT
        parquet-cpp-feedstock/ci_support/run_docker_build.sh
        cp parquet-cpp-feedstock/build_artefacts/linux-64/parquet-cpp*.tar.bz2 $(dir $@)
        touch $@

./artifacts/arrow-cpp/BUILT:
        arrow-cpp-feedstock/ci_support/run_docker_build.sh
        cp arrow-cpp-feedstock/build_artefacts/linux-64/arrow-cpp*.tar.bz2 $(dir $@)
        touch $@
```
