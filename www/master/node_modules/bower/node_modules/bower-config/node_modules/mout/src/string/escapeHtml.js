define(['../lang/toString'], function(toString) {

    /**
     * Escapes a string for insertion into HTML.
     */
    function escapeHtml(str){
        str = toString(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/'/g, '&#39;')
            .replace(/"/g, '&quot;');
        return str;
    }

    return escapeHtml;

});
