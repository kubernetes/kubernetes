

    /**
     * Returns a function that call a method on the passed object
     */
    function func(name){
        return function(obj){
            return obj[name]();
        };
    }

    module.exports = func;


