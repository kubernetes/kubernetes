var _ = require('underscore');
var fs = require('fs');
var yaml = require('js-yaml');
var colors = require('colors/safe');

var write_cloud_config_from_object = function (data, output_file) {
  try {
    fs.writeFileSync(output_file, [
      '#cloud-config',
      yaml.safeDump(data),
    ].join("\n"));
    return output_file;
  } catch (e) {
    console.log(colors.red(e));
  }
};

exports.generate_environment_file_entry_from_object = function (hostname, environ) {
  var data = {
    hostname: hostname,
    environ_array: _.map(environ, function (value, key) {
      return [key.toUpperCase(), JSON.stringify(value.toString())].join('=');
    }),
  };

  return {
    permissions: '0600',
    owner: 'root',
    content: _.template("<%= environ_array.join('\\n') %>\n")(data),
    path: _.template("/etc/weave.<%= hostname %>.env")(data),
  };
};

exports.process_template = function (input_file, output_file, processor) {
  var data = {};
  try {
    data = yaml.safeLoad(fs.readFileSync(input_file, 'utf8'));
  } catch (e) {
    console.log(colors.red(e));
  }
  return write_cloud_config_from_object(processor(_.clone(data)), output_file);
};

exports.write_files_from = function (local_dir, remote_dir) {
  try {
    return _.map(fs.readdirSync(local_dir), function (fn) {
      return {
        path: [remote_dir, fn].join('/'),
        owner: 'root',
        permissions: '0640',
        encoding: 'base64',
        content: fs.readFileSync([local_dir, fn].join('/')).toString('base64'),
      };
    });
  } catch (e) {
    console.log(colors.red(e));
  }
};
