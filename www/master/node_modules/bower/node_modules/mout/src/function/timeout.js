define(['../array/slice'], function (slice) {

    /**
     * Delays the call of a function within a given context.
     */
    function timeout(fn, millis, context){

        var args = slice(arguments, 3);

        return setTimeout(function() {
            fn.apply(context, args);
        }, millis);
    }

    return timeout;

});
