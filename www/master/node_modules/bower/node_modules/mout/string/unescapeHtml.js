var toString = require('../lang/toString');

    /**
     * Unescapes HTML special chars
     */
    function unescapeHtml(str){
        str = toString(str)
            .replace(/&amp;/g , '&')
            .replace(/&lt;/g  , '<')
            .replace(/&gt;/g  , '>')
            .replace(/&#0*39;/g , "'")
            .replace(/&quot;/g, '"');
        return str;
    }

    module.exports = unescapeHtml;


