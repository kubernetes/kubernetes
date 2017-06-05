module.exports = function(hljs) {
  var BUILTIN_MODULES =
    'ObjectLoader Animate MovieCredits Slides Filters Shading Materials LensFlare Mapping VLCAudioVideo ' +
    'StereoDecoder PointCloud NetworkAccess RemoteControl RegExp ChromaKey Snowfall NodeJS Speech Charts';

  var XL_KEYWORDS = {
    keyword:
      'if then else do while until for loop import with is as where when by data constant ' +
      'integer real text name boolean symbol infix prefix postfix block tree',
    literal:
      'true false nil',
    built_in:
      'in mod rem and or xor not abs sign floor ceil sqrt sin cos tan asin ' +
      'acos atan exp expm1 log log2 log10 log1p pi at text_length text_range ' +
      'text_find text_replace contains page slide basic_slide title_slide ' +
      'title subtitle fade_in fade_out fade_at clear_color color line_color ' +
      'line_width texture_wrap texture_transform texture scale_?x scale_?y ' +
      'scale_?z? translate_?x translate_?y translate_?z? rotate_?x rotate_?y ' +
      'rotate_?z? rectangle circle ellipse sphere path line_to move_to ' +
      'quad_to curve_to theme background contents locally time mouse_?x ' +
      'mouse_?y mouse_buttons ' +
      BUILTIN_MODULES
  };

  var DOUBLE_QUOTE_TEXT = {
    className: 'string',
    begin: '"', end: '"', illegal: '\\n'
  };
  var SINGLE_QUOTE_TEXT = {
    className: 'string',
    begin: '\'', end: '\'', illegal: '\\n'
  };
  var LONG_TEXT = {
    className: 'string',
    begin: '<<', end: '>>'
  };
  var BASED_NUMBER = {
    className: 'number',
    begin: '[0-9]+#[0-9A-Z_]+(\\.[0-9-A-Z_]+)?#?([Ee][+-]?[0-9]+)?'
  };
  var IMPORT = {
    beginKeywords: 'import', end: '$',
    keywords: XL_KEYWORDS,
    contains: [DOUBLE_QUOTE_TEXT]
  };
  var FUNCTION_DEFINITION = {
    className: 'function',
    begin: /[a-z][^\n]*->/, returnBegin: true, end: /->/,
    contains: [
      hljs.inherit(hljs.TITLE_MODE, {starts: {
        endsWithParent: true,
        keywords: XL_KEYWORDS
      }})
    ]
  };
  return {
    aliases: ['tao'],
    lexemes: /[a-zA-Z][a-zA-Z0-9_?]*/,
    keywords: XL_KEYWORDS,
    contains: [
    hljs.C_LINE_COMMENT_MODE,
    hljs.C_BLOCK_COMMENT_MODE,
    DOUBLE_QUOTE_TEXT,
    SINGLE_QUOTE_TEXT,
    LONG_TEXT,
    FUNCTION_DEFINITION,
    IMPORT,
    BASED_NUMBER,
    hljs.NUMBER_MODE
    ]
  };
};