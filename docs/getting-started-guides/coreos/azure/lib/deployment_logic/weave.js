var _ = require('underscore');

var util = require('../util.js');
var cloud_config = require('../cloud_config.js');

var write_basic_weave_cluster_cloud_config = function (env_files) {
  var input_file = './cloud_config_templates/basic-weave-cluster-template.yml';
  var output_file = util.join_output_file_path('basic-weave-cluster', 'generated.yml');

  return cloud_config.process_template(input_file, output_file, function(data) {
    data.write_files = env_files;
    return data;
  });
};

exports.create_basic_cloud_config = function (node_count, conf) {
  var elected_node = 0;

  var make_node_config = function (n) {
    return cloud_config.generate_environment_file_entry_from_object(util.hostname(n), {
      weavedns_addr: util.ipv4([10, 10, 1, 10+n], 24),
      weave_password: conf.weave_salt,
      weave_peers: n === elected_node ? "" : util.hostname(elected_node),
    });
  };

  return write_basic_weave_cluster_cloud_config(_(node_count).times(make_node_config));
};

