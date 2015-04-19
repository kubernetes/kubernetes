var extensions = {
  '.cjsx': 'node-cjsx/register',
  '.co': 'coco',
  '.coffee': 'coffee-script/register',
  '.coffee.md': 'coffee-script/register',
  '.csv': 'require-csv',
  '.iced': 'iced-coffee-script/register',
  '.iced.md': 'iced-coffee-script/register',
  '.ini': 'require-ini',
  '.js': null,
  '.json': null,
  '.json5': 'json5/lib/require',
  '.jsx': 'node-jsx',
  '.litcoffee': 'coffee-script/register',
  '.liticed': 'iced-coffee-script/register',
  '.ls': 'LiveScript',
  '.toml': 'toml-require',
  '.ts': 'typescript-require',
  '.xml': 'require-xml',
  '.yaml': 'require-yaml',
  '.yml': 'require-yaml'
};

var register = {
  'node-jsx': function (module) {
    module.install({ extension: '.jsx', harmony: true });
  },
  'toml-require': function (module) {
    module.install();
  }
};

var jsVariantExtensions = [
  '.js',
  '.cjsx',
  '.co',
  '.coffee',
  '.coffee.md',
  '.iced',
  '.iced.md',
  '.jsx',
  '.litcoffee',
  '.liticed',
  '.ls',
  '.ts'
];

module.exports = {
  extensions: extensions,
  register: register,
  jsVariants: jsVariantExtensions.reduce(function (result, ext) {
    result[ext] = extensions[ext];
    return result;
  }, {})
};
