define(function () {

    /**
     * Returns a function that call a method on the passed object
     */
    function func(name){
        return function(obj){
            return obj[name]();
        };
    }

    return func;

});
