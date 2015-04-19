define(['../lang/toString', './repeat'], function(toString, repeat) {

    /**
     * Pad string with `char` if its' length is smaller than `minLen`
     */
    function lpad(str, minLen, ch) {
        str = toString(str);
        ch = ch || ' ';

        return (str.length < minLen) ?
            repeat(ch, minLen - str.length) + str : str;
    }

    return lpad;

});
