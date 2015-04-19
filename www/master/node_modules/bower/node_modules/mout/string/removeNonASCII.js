var toString = require('../lang/toString');
    /**
     * Remove non-printable ASCII chars
     */
    function removeNonASCII(str){
        str = toString(str);

        // Matches non-printable ASCII chars -
        // http://en.wikipedia.org/wiki/ASCII#ASCII_printable_characters
        return str.replace(/[^\x20-\x7E]/g, '');
    }

    module.exports = removeNonASCII;

