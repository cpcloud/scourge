# scourge

Build arbitrary conda dependency graphs from source


## Example

Here's an example of how to use `scourge`.


### 1. Initialization: `scourge init`

This clones the latest ``condaforge/linux-anvil`` docker image and sets up
feedstocks provided at the command line.

The ``@`` allows arbitrary git ref syntax for each package.

```sh
$ scourge init arrow-cpp@master parquet-cpp@master pyarrow@master                            
Using default tag: latest                      
latest: Pulling from condaforge/linux-anvil    
Digest: sha256:dd53ef1792f777fa8b2abf30221bf9beb6cb6a81d158d498f0219c4d475696cd                
Status: Image is up to date for condaforge/linux-anvil:latest                                  
Cloning feedstocks: 100%|█████████████████████████████████████| 3/3 [00:00<00:00, 11.65it/s]
```

```sh
$ scourge init -h
Usage: scourge init [OPTIONS] [PACKAGE_SPECIFICATIONS]...

  Pull in the conda forge docker image and clone feedstocks.

Options:
  -i, --image TEXT               The Docker image to use to build packages.
  -a, --artifact-directory PATH  The location to place tarballs upon a
                                 successful build.
  -h, --help                     Show this message and exit.
```

### 2. Package builds: `scourge build`

This step fires up a bunch of docker containers--one for each build--and writes
the build log for each one to a file in the ``$PWD/logs`` directory.


```sh
$ scourge build -h
Usage: scourge build [OPTIONS]

  Build conda forge packages in parallel

Options:
  -c, --constraints TEXT  Additional special software constraints - e.g.,
                          numpy/python.
  -e, --environment TEXT  Additional environment variables to pass to builds
  -j, --jobs INTEGER      Number of workers to use for building
  -h, --help              Show this message and exit.
```


### * `scourge clean`

This is useful if you want to start from scratch. It removes everything in the
directory where you ran `scourge init`/`scourge build`.
