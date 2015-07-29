var _ = require('underscore');
_.mixin(require('underscore.string').exports());

var util = require('../util.js');
var cloud_config = require('../cloud_config.js');


etcd_initial_cluster_conf_self = function (conf) {
  var port = '2380';

  var data = {
    nodes: _(conf.nodes.etcd).times(function (n) {
      var host = util.hostname(n, 'etcd');
      return [host, [host, port].join(':')].join('=http://');
    }),
  };

  return {
    'name': 'etcd2.service',
    'drop-ins': [{
      'name': '50-etcd-initial-cluster.conf',
      'content': _.template("[Service]\nEnvironment=ETCD_INITIAL_CLUSTER=<%= nodes.join(',') %>\n")(data),
    }],
  };
};

etcd_initial_cluster_conf_kube = function (conf) {
  var port = '4001';

  var data = {
    nodes: _(conf.nodes.etcd).times(function (n) {
      var host = util.hostname(n, 'etcd');
      return 'http://' + [host, port].join(':');
    }),
  };

  return {
    'name': 'kube-apiserver.service',
    'drop-ins': [{
      'name': '50-etcd-initial-cluster.conf',
      'content': _.template("[Service]\nEnvironment=ETCD_SERVERS=--etcd_servers=<%= nodes.join(',') %>\n")(data),
    }],
  };
};

exports.create_etcd_cloud_config = function (node_count, conf) {
  var input_file = './cloud_config_templates/kubernetes-cluster-etcd-node-template.yml';
  var output_file = util.join_output_file_path('kubernetes-cluster-etcd-nodes', 'generated.yml');

  return cloud_config.process_template(input_file, output_file, function(data) {
    data.coreos.units.push(etcd_initial_cluster_conf_self(conf));
    return data;
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

  var write_files_extra = cloud_config.write_files_from('addons', '/etc/kubernetes/addons');
  return cloud_config.process_template(input_file, output_file, function(data) {
    data.write_files = data.write_files.concat(_(node_count).times(make_node_config), write_files_extra);
    data.coreos.units.push(etcd_initial_cluster_conf_kube(conf));
    return data;
  });
};
