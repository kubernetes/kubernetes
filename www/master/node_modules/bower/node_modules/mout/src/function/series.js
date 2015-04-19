define(function () {

    /**
     * Returns a function that will execute a list of functions in sequence
     * passing the same arguments to each one. (useful for batch processing
     * items during a forEach loop)
     */
    function series(){
        var fns = arguments;
        return function(){
            var i = 0,
                n = fns.length;
            while (i < n) {
                fns[i].apply(this, arguments);
                i += 1;
            }
        };
    }

    return series;

});
