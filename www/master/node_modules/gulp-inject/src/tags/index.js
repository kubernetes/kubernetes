
/**
 * Constants
 */
var DEFAULT_NAME = 'inject';
var DEFAULT_TARGET = 'html';
var DEFAULTS = {
  STARTS: {
    'html': '<!-- {{name}}:{{ext}} -->',
    'jsx': '{/* {{name}}:{{ext}} */}',
    'jade': '//- {{name}}:{{ext}}',
    'slm': '/ {{name}}:{{ext}}',
    'haml': '-# {{name}}:{{ext}}'
  },
  ENDS: {
    'html': '<!-- endinject -->',
    'jsx': '{/* endinject */}',
    'jade': '//- endinject',
    'slm': '/ endinject',
    'haml': '-# endinject'
  }
};

module.exports = function tags () {
  var tags = {
    name: DEFAULT_NAME
  };

  tags.start = getTag.bind(tags, DEFAULTS.STARTS);
  tags.end = getTag.bind(tags, DEFAULTS.ENDS);

  return tags;
};

function getTag (defaults, targetExt, sourceExt, defaultValue) {
  var tag = defaultValue;
  if (!tag) {
    tag = defaults[targetExt] || defaults[DEFAULT_TARGET];
  } else if (typeof tag === 'function') {
    tag = tag(targetExt, sourceExt);
  }
  if (!tag) {
    return;
  }
  tag = tag.replace(new RegExp(escapeForRegExp('{{ext}}'), 'g'), sourceExt);
  return tag.replace(new RegExp(escapeForRegExp('{{name}}'), 'g'), this.name);
}

function escapeForRegExp (str) {
  return str.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
}
