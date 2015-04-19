define(['../array/slice'], function (slice) {

    /**
     * Creates a partially applied function.
     */
    function partial(f) {
        var as = slice(arguments, 1);
        return function() {
            var args = as.concat(slice(arguments));
            for (var i = args.length; i--;) {
                if (args[i] === partial._) {
                    args[i] = args.splice(-1)[0];
                }
            }
            return f.apply(this, args);
        };
    }

    partial._ = {};

    return partial;

});
