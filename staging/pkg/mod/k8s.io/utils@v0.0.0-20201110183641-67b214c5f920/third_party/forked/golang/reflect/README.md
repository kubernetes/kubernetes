This was originally forked from https://github.com/golang/go/blob/master/src/reflect/deepequal.go in order to

- consider empty lists and empty maps equal to their nil counterparts
- add a `AddFuncs` mechanism to add custom equality funcs for specific types. 

Meanwhile it has diverged quite a lot while still following the original algorithm at the core though.
