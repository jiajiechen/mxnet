# Read MXNet Code
- All of the module interfaces are listed in [include](https://github.com/dmlc/mxnet/tree/master/include/mxnet). The
  interfaces are heavily documented.
- Read the
  [Doxygen Version](http://mxnet.io/api/c++/index.html) of the document.
- Each module depends on other modules as defined in the header files in
  [include](https://github.com/dmlc/mxnet/tree/master/include/mxnet).
- Module implementation is in the  [src](https://github.com/dmlc/mxnet/tree/master/src) folder.
- Source code sees only the file within its folder,
  [src/common](https://github.com/dmlc/mxnet/tree/master/src/common) and [include](https://github.com/dmlc/mxnet/tree/master/include/mxnet).

Most modules are mostly self-contained, with interface dependency on the engine.
You're free to pick the one you are interested in and read that part.

## Next Steps

* [Develop and Hack MXNet](http://mxnet.io/how_to/develop_and_hack.html)
