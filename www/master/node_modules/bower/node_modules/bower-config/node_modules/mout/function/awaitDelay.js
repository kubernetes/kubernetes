var now = require('../time/now');
var timeout = require('./timeout');
var append = require('../array/append');

    /**
     * Ensure a minimum delay for callbacks
     */
    function awaitDelay( callback, delay ){
        var baseTime = now() + delay;
        return function() {
            // ensure all browsers will execute it asynchronously (avoid hard
            // to catch errors) not using "0" because of old browsers and also
            // since new browsers increase the value to be at least "4"
            // http://www.whatwg.org/specs/web-apps/current-work/multipage/timers.html#dom-windowtimers-settimeout
            var ms = Math.max(baseTime - now(), 4);
            return timeout.apply(this, append([callback, ms, this], arguments));
        };
    }

    module.exports = awaitDelay;


