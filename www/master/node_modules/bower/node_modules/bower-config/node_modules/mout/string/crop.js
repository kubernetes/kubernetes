var toString = require('../lang/toString');
var truncate = require('./truncate');
    /**
     * Truncate string at full words.
     */
     function crop(str, maxChars, append) {
         str = toString(str);
         return truncate(str, maxChars, append, true);
     }

     module.exports = crop;

