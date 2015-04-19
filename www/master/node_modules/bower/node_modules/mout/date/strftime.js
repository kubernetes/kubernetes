var pad = require('../number/pad');
var lpad = require('../string/lpad');
var i18n = require('./i18n_');
var dayOfTheYear = require('./dayOfTheYear');
var timezoneOffset = require('./timezoneOffset');
var timezoneAbbr = require('./timezoneAbbr');
var weekOfTheYear = require('./weekOfTheYear');

    var _combinations = {
        'D': '%m/%d/%y',
        'F': '%Y-%m-%d',
        'r': '%I:%M:%S %p',
        'R': '%H:%M',
        'T': '%H:%M:%S',
        'x': 'locale',
        'X': 'locale',
        'c': 'locale'
    };


    /**
     * format date based on strftime format
     */
    function strftime(date, format, localeData){
        localeData = localeData  || i18n;
        var reToken = /%([a-z%])/gi;

        function makeIterator(fn) {
            return function(match, token){
                return fn(date, token, localeData);
            };
        }

        return format
            .replace(reToken, makeIterator(expandCombinations))
            .replace(reToken, makeIterator(convertToken));
    }


    function expandCombinations(date, token, l10n){
        if (token in _combinations) {
            var expanded = _combinations[token];
            return expanded === 'locale'? l10n[token] : expanded;
        } else {
            return '%'+ token;
        }
    }


    function convertToken(date, token, l10n){
        switch (token){
            case 'a':
                return l10n.days_abbr[date.getDay()];
            case 'A':
                return l10n.days[date.getDay()];
            case 'h':
            case 'b':
                return l10n.months_abbr[date.getMonth()];
            case 'B':
                return l10n.months[date.getMonth()];
            case 'C':
                return pad(Math.floor(date.getFullYear() / 100), 2);
            case 'd':
                return pad(date.getDate(), 2);
            case 'e':
                return pad(date.getDate(), 2, ' ');
            case 'H':
                return pad(date.getHours(), 2);
            case 'I':
                return pad(date.getHours() % 12, 2);
            case 'j':
                return pad(dayOfTheYear(date), 3);
            case 'l':
                return lpad(date.getHours() % 12, 2);
            case 'L':
                return pad(date.getMilliseconds(), 3);
            case 'm':
                return pad(date.getMonth() + 1, 2);
            case 'M':
                return pad(date.getMinutes(), 2);
            case 'n':
                return '\n';
            case 'p':
                return date.getHours() >= 12? l10n.pm : l10n.am;
            case 'P':
                return convertToken(date, 'p', l10n).toLowerCase();
            case 's':
                return date.getTime() / 1000;
            case 'S':
                return pad(date.getSeconds(), 2);
            case 't':
                return '\t';
            case 'u':
                var day = date.getDay();
                return day === 0? 7 : day;
            case 'U':
                return pad(weekOfTheYear(date), 2);
            case 'w':
                return date.getDay();
            case 'W':
                return pad(weekOfTheYear(date, 1), 2);
            case 'y':
                return pad(date.getFullYear() % 100, 2);
            case 'Y':
                return pad(date.getFullYear(), 4);
            case 'z':
                return timezoneOffset(date);
            case 'Z':
                return timezoneAbbr(date);
            case '%':
                return '%';
            default:
                // keep unrecognized tokens
                return '%'+ token;
        }
    }


    module.exports = strftime;


