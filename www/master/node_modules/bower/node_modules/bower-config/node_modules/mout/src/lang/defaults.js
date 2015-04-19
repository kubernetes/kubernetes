define(['./toArray', '../array/find'], function (toArray, find) {

    /**
     * Return first non void argument
     */
    function defaults(var_args){
        return find(toArray(arguments), nonVoid);
    }

    function nonVoid(val){
        return val != null;
    }

    return defaults;

});
