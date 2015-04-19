var some = require('../array/some');

    var datePatterns = [
        /^([0-9]{4})$/,                        // YYYY
        /^([0-9]{4})-([0-9]{2})$/,             // YYYY-MM (YYYYMM not allowed)
        /^([0-9]{4})-?([0-9]{2})-?([0-9]{2})$/ // YYYY-MM-DD or YYYYMMDD
    ];
    var ORD_DATE = /^([0-9]{4})-?([0-9]{3})$/; // YYYY-DDD

    var timePatterns = [
        /^([0-9]{2}(?:\.[0-9]*)?)$/,                      // HH.hh
        /^([0-9]{2}):?([0-9]{2}(?:\.[0-9]*)?)$/,          // HH:MM.mm
        /^([0-9]{2}):?([0-9]{2}):?([0-9]{2}(\.[0-9]*)?)$/ // HH:MM:SS.ss
    ];

    var DATE_TIME = /^(.+)T(.+)$/;
    var TIME_ZONE = /^(.+)([+\-])([0-9]{2}):?([0-9]{2})$/;

    function matchAll(str, patterns) {
        var match;
        var found = some(patterns, function(pattern) {
            return !!(match = pattern.exec(str));
        });

        return found ? match : null;
    }

    function getDate(year, month, day) {
        var date = new Date(Date.UTC(year, month, day));

        // Explicitly set year to avoid Date.UTC making dates < 100 relative to
        // 1900
        date.setUTCFullYear(year);

        var valid =
            date.getUTCFullYear() === year &&
            date.getUTCMonth() === month &&
            date.getUTCDate() === day;
        return valid ? +date : NaN;
    }

    function parseOrdinalDate(str) {
        var match = ORD_DATE.exec(str);
        if (match ) {
            var year = +match[1],
                day = +match[2],
                date = new Date(Date.UTC(year, 0, day));

            if (date.getUTCFullYear() === year) {
                return +date;
            }
        }

        return NaN;
    }

    function parseDate(str) {
        var match, year, month, day;

        match = matchAll(str, datePatterns);
        if (match === null) {
            // Ordinal dates are verified differently.
            return parseOrdinalDate(str);
        }

        year = (match[1] === void 0) ? 0 : +match[1];
        month = (match[2] === void 0) ? 0 : +match[2] - 1;
        day = (match[3] === void 0) ? 1 : +match[3];

        return getDate(year, month, day);
    }

    function getTime(hr, min, sec) {
        var valid =
            (hr < 24 && hr >= 0 &&
             min < 60 && min >= 0 &&
             sec < 60 && min >= 0) ||
            (hr === 24 && min === 0 && sec === 0);
        if (!valid) {
            return NaN;
        }

        return ((hr * 60 + min) * 60 + sec) * 1000;
    }

    function parseOffset(str) {
        var match;
        if (str.charAt(str.length - 1) === 'Z') {
            str = str.substring(0, str.length - 1);
        } else {
            match = TIME_ZONE.exec(str);
            if (match) {
                var hours = +match[3],
                    minutes = (match[4] === void 0) ? 0 : +match[4],
                    offset = getTime(hours, minutes, 0);

                if (match[2] === '-') {
                    offset *= -1;
                }

                return { offset: offset, time: match[1] };
            }
        }

        // No time zone specified, assume UTC
        return { offset: 0, time: str };
    }

    function parseTime(str) {
        var match;
        var offset = parseOffset(str);

        str = offset.time;
        offset = offset.offset;
        if (isNaN(offset)) {
            return NaN;
        }

        match = matchAll(str, timePatterns);
        if (match === null) {
            return NaN;
        }

        var hours = (match[1] === void 0) ? 0 : +match[1],
            minutes = (match[2] === void 0) ? 0 : +match[2],
            seconds = (match[3] === void 0) ? 0 : +match[3];

        return getTime(hours, minutes, seconds) - offset;
    }

    /**
     * Parse an ISO8601 formatted date string, and return a Date object.
     */
    function parseISO8601(str){
        var match = DATE_TIME.exec(str);
        if (!match) {
            // No time specified
            return parseDate(str);
        }

        return parseDate(match[1]) + parseTime(match[2]);
    }

    module.exports = parseISO8601;


