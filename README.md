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
$ scourge gen
all: ./artifacts/pyarrow ./artifacts/parquet-cpp ./artifacts/arrow-cpp

clean:
        rm -rf ./artifacts

./artifacts/pyarrow: ./artifacts/arrow-cpp ./artifacts/parquet-cpp
        pyarrow-feedstock/ci_support/run_docker_build.sh
        cp $(wildcard pyarrow-feedstock/build_artefacts/pyarrow*.tar.bz2) $@


./artifacts/parquet-cpp: ./artifacts/arrow-cpp
        parquet-cpp-feedstock/ci_support/run_docker_build.sh
        cp $(wildcard parquet-cpp-feedstock/build_artefacts/parquet-cpp*.tar.bz2) $@


./artifacts/arrow-cpp: 
        arrow-cpp-feedstock/ci_support/run_docker_build.sh
        cp $(wildcard arrow-cpp-feedstock/build_artefacts/arrow-cpp*.tar.bz2) $@
```
