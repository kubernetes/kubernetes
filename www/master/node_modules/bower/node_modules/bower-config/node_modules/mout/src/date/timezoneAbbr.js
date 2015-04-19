define(['./timezoneOffset'], function(timezoneOffset) {

    /**
     * Abbreviated time zone name or similar information.
     */
    function timezoneAbbr(date){
        // Date.toString gives different results depending on the
        // browser/system so we fallback to timezone offset
        // chrome: 'Mon Apr 08 2013 09:02:04 GMT-0300 (BRT)'
        // IE: 'Mon Apr 8 09:02:04 UTC-0300 2013'
        var tz = /\(([A-Z]{3,4})\)/.exec(date.toString());
        return tz? tz[1] : timezoneOffset(date);
    }

    return timezoneAbbr;

});
