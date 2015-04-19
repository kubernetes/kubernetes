var toString = require('../lang/toString');
var repeat = require('./repeat');

    /**
     * Pad string with `char` if its' length is smaller than `minLen`
     */
    function lpad(str, minLen, ch) {
        str = toString(str);
        ch = ch || ' ';

        return (str.length < minLen) ?
            repeat(ch, minLen - str.length) + str : str;
    }

    module.exports = lpad;


