var now = require('../time/now');

    /**
     */
    function throttle(fn, delay){
        var context, timeout, result, args,
            diff, prevCall = 0;
        function delayed(){
            prevCall = now();
            timeout = null;
            result = fn.apply(context, args);
        }
        function throttled(){
            context = this;
            args = arguments;
            diff = delay - (now() - prevCall);
            if (diff <= 0) {
                clearTimeout(timeout);
                delayed();
            } else if (! timeout) {
                timeout = setTimeout(delayed, diff);
            }
            return result;
        }
        throttled.cancel = function(){
            clearTimeout(timeout);
        };
        return throttled;
    }

    module.exports = throttle;


