var _ = require('underscore');
_.mixin(require('underscore.string').exports());

var util = require('../util.js');
var cloud_config = require('../cloud_config.js');


exports.create_etcd_cloud_config = function (node_count, conf) {
  var input_file = './cloud_config_templates/kubernetes-cluster-etcd-node-template.yml';

  var peers = [ ];
  for (var i = 0; i < node_count; i++) {
    peers.push(util.hostname(i, 'etcd') + '=http://' + util.hostname(i, 'etcd') + ':2380');
  }
  var cluster = peers.join(',');

  return _(node_count).times(function (n) {
    var output_file = util.join_output_file_path('kubernetes-cluster-etcd-node-' + n, 'generated.yml');

    return cloud_config.process_template(input_file, output_file, function(data) {
      for (var i = 0; i < data.coreos.units.length; i++) {
        var unit = data.coreos.units[i];
        if (unit.name === 'etcd2.service') {
          unit.content = _.replaceAll(_.replaceAll(unit.content, '%host%', util.hostname(n, 'etcd')), '%cluster%', cluster);
          break;
        }
      }
      return data;
    });
  });
};

exports.create_node_cloud_config = function (node_count, conf) {
  var elected_node = 0;

  var input_file = './cloud_config_templates/kubernetes-cluster-main-nodes-template.yml';
  var output_file = util.join_output_file_path('kubernetes-cluster-main-nodes', 'generated.yml');

  var make_node_config = function (n) {
    return cloud_config.generate_environment_file_entry_from_object(util.hostname(n, 'kube'), {
      weave_password: conf.weave_salt,
      weave_peers: n === elected_node ? "" : util.hostname(elected_node, 'kube'),
      breakout_route: util.ipv4([10, 2, 0, 0], 16),
      bridge_address_cidr: util.ipv4([10, 2, n, 1], 24),
    });
  };
  return cloud_config.process_template(input_file, output_file, function(data) {
    data.write_files = data.write_files.concat(_(node_count).times(make_node_config));
    return data;
  });
};
