

    /**
     * Returns a new function that will return the value
     */
    function constant(value){
        return function() {
            return value;
        };
    }

    module.exports = constant;


