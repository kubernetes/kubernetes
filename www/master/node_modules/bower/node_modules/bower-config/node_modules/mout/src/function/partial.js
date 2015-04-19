define(['../array/slice'], function (slice) {

    /**
     * Creates a partially applied function.
     */
    function partial(fn, var_args){
        var argsArr = slice(arguments, 1); //curried args
        return function(){
            return fn.apply(this, argsArr.concat(slice(arguments)));
        };
    }

    return partial;

});
