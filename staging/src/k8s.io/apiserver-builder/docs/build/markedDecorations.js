;(function() {
    var brodocDec = {};

    var navIds = [];
    var bodyContent = '';
    var codeTabs = [];

    var idAffix = 0;
    var uniqueNav = [];

    brodocDec.decorateMarked = function(renderer) {
        renderer.heading = (text, level, raw) => {
            var id = raw.toLowerCase().replace(/[^\w]+/g, '-');
            if ((uniqueNav.indexOf(id) !== -1) && (level === 2)) {
                idAffix++;
                id += '-' + idAffix;
            } else {
                uniqueNav.push(id);
            }
            if (level < 3) {
                navIds.push(
                    {
                        id: id,
                        text: text,
                        level: level
                    }
                );
            }
            return '<h'
                + level
                + ' id="'
                + renderer.options.headerPrefix
                + id
                + '">'
                + text
                + '</h'
                + level
                + '>\n';
        };

        renderer.blockquote = function(quote) {
            var bdregex = /(bdocs-tab:)[^\s]*/;
            var bdoc = quote.match(bdregex);
            if (bdoc) {
                var bdocTab = bdoc[0].split(':')[1];
                var bdquote = quote.replace(bdoc[0], '');
                return '<blockquote class="code-block ' + bdocTab + '">\n' + bdquote + '</blockquote>\n';
            } else {
                return '<blockquote>\n' + quote + '</blockquote>\n';
            }
        };

        renderer.code = function (code, lang, escaped) {
            var bdocGroup = lang.substring(0, lang.indexOf('_'));
            var bdocTab = bdocGroup.split(':')[1];
            var hlang = lang.substring(lang.indexOf('_')+1);

            if (renderer.options.highlight) {
                var out = renderer.options.highlight(code, hlang);
                if (out !== null && out !== code) {
                    escaped = true;
                    code = out;
                }
            }

            var tabLang = hlang ? hlang : 'generic';
            if (codeTabs.indexOf(bdocTab) === -1) {
                codeTabs.push(bdocTab);
            }

            if (!hlang) {
                return '<pre class="code-block"><code class="generic">'
                    + (escaped ? code : escape(code, true))
                    + '\n</code></pre>';
            }

            return '<pre class="code-block '
                + bdocTab
                + '"><code class="'
                + renderer.options.langPrefix
                + escape(hlang, true)
                + '">'
                + (escaped ? code : escape(code, true))
                + '\n</code></pre>\n';
        };
    };
    
    if (typeof module !== 'undefined' && typeof exports === 'object') {
        module.exports = brodocDec;
    } else if (typeof define === 'function' && define.amd) {
        define(function() { return brodocDec; });
    } else {
        this.brodocDec = brodocDec;
    }
    brodocDec.navIds = navIds;
    brodocDec.codeTabs = codeTabs;
    return brodocDec;

}).call(function() {
  return this || (typeof window !== 'undefined' ? window : global);
}());