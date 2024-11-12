# Cognifit Benchmark

### Vision
- user only needs to create a model class (subclass of benchmark class which forces to overwride a method that gets list/dict of the class variables)
  - everything else, including training pipline, data loading, parsing, cross validation etc. is then handled automatically  
- this can then be used by the model parsing and model util without needing the user to change those files
- also the model options are automatically determined by the classes inside the model folder


### License
This code is distributed under the [MIT LICENSE](LICENSE).

