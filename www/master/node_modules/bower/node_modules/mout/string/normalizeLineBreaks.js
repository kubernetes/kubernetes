var toString = require('../lang/toString');

    /**
     * Convert line-breaks from DOS/MAC to a single standard (UNIX by default)
     */
    function normalizeLineBreaks(str, lineEnd) {
        str = toString(str);
        lineEnd = lineEnd || '\n';

        return str
            .replace(/\r\n/g, lineEnd) // DOS
            .replace(/\r/g, lineEnd)   // Mac
            .replace(/\n/g, lineEnd);  // Unix
    }

    module.exports = normalizeLineBreaks;


