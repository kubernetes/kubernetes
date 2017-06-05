module.exports = function(hljs) {
  var BACKTICK_ESCAPE = {
    begin: /`[\s\S]/
  };

  return {
    case_insensitive: true,
    keywords: {
      keyword: 'Break Continue Else Gosub If Loop Return While',
      literal: 'A|0 true false NOT AND OR',
      built_in: 'ComSpec Clipboard ClipboardAll ErrorLevel',
    },
    contains: [
      {
        className: 'built_in',
        begin: 'A_[a-zA-Z0-9]+'
      },
      BACKTICK_ESCAPE,
      hljs.inherit(hljs.QUOTE_STRING_MODE, {contains: [BACKTICK_ESCAPE]}),
      hljs.COMMENT(';', '$', {relevance: 0}),
      {
        className: 'number',
        begin: hljs.NUMBER_RE,
        relevance: 0
      },
      {
        className: 'variable', // FIXME
        begin: '%', end: '%',
        illegal: '\\n',
        contains: [BACKTICK_ESCAPE]
      },
      {
        className: 'symbol',
        contains: [BACKTICK_ESCAPE],
        variants: [
          {begin: '^[^\\n";]+::(?!=)'},
          {begin: '^[^\\n";]+:(?!=)', relevance: 0} // zero relevance as it catches a lot of things
                                                    // followed by a single ':' in many languages
        ]
      },
      {
        // consecutive commas, not for highlighting but just for relevance
        begin: ',\\s*,'
      }
    ]
  }
};