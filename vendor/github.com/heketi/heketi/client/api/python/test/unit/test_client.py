#
# Copyright (c) 2015 The heketi Authors
#
# This file is licensed to you under your choice of the GNU Lesser
# General Public License, version 3 or any later version (LGPLv3 or
# later), as published by the Free Software Foundation,
# or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
# http://www.apache.org/licenses/LICENSE-2.0>.
#
# You may not use this file except in compliance with those terms.
#

import unittest
import requests
from heketi import HeketiClient


TEST_ADMIN_KEY = "My Secret"
TEST_SERVER = "http://localhost:8080"


class test_heketi(unittest.TestCase):

    def test_cluster(self):
        c = HeketiClient(TEST_SERVER, "admin", TEST_ADMIN_KEY)

        cluster = c.cluster_create()
        self.assertEqual(True, cluster['id'] != "")
        self.assertEqual(True, len(cluster['nodes']) == 0)
        self.assertEqual(True, len(cluster['volumes']) == 0)

        # Request bad id
        with self.assertRaises(requests.exceptions.HTTPError):
            c.cluster_info("bad")

        # Get info about the cluster
        info = c.cluster_info(cluster['id'])
        self.assertEqual(True, info == cluster)

        # Get a list of clusters
        list = c.cluster_list()
        self.assertEqual(True, len(list['clusters']) == 1)
        self.assertEqual(True, list['clusters'][0] == cluster['id'])

        # Delete non-existent cluster
        with self.assertRaises(requests.exceptions.HTTPError):
            c.cluster_delete("badid")

        # Delete current cluster
        self.assertEqual(True, c.cluster_delete(info['id']))

    def test_node(self):
        node_req = {}

        c = HeketiClient(TEST_SERVER, "admin", TEST_ADMIN_KEY)
        self.assertEqual(True, c != '')

        # Create cluster
        cluster = c.cluster_create()
        self.assertEqual(True, cluster['id'] != "")
        self.assertEqual(True, len(cluster['nodes']) == 0)
        self.assertEqual(True, len(cluster['volumes']) == 0)

        # Add node to unknown cluster
        node_req['cluster'] = "bad_id"
        node_req['zone'] = 10
        node_req['hostnames'] = {
            "manage": ["node1-manage.gluster.lab.com"],
            "storage": ["node1-storage.gluster.lab.com"]
        }

        with self.assertRaises(requests.exceptions.HTTPError):
            c.node_add(node_req)

        # Create node request packet
        node_req['cluster'] = cluster['id']
        node = c.node_add(node_req)
        self.assertEqual(True, node['zone'] == node_req['zone'])
        self.assertEqual(True, node['id'] != "")
        self.assertEqual(True, node_req['hostnames'] == node['hostnames'])
        self.assertEqual(True, len(node['devices']) == 0)

        # Info on invalid id
        with self.assertRaises(requests.exceptions.HTTPError):
            c.node_info("badid")

        # Get node info
        info = c.node_info(node['id'])
        self.assertEqual(True, info == node)
        self.assertEqual(info['state'], 'online')

        # Set offline
        state = {}
        state['state'] = 'offline'
        self.assertEqual(True, c.node_state(node['id'], state))

        # Get node info
        info = c.node_info(node['id'])
        self.assertEqual(info['state'], 'offline')

        state['state'] = 'online'
        self.assertEqual(True, c.node_state(node['id'], state))

        info = c.node_info(node['id'])
        self.assertEqual(info['state'], 'online')

        # Delete invalid node
        with self.assertRaises(requests.exceptions.HTTPError):
            c.node_delete("badid")

        # Can't delete cluster with a node
        with self.assertRaises(requests.exceptions.HTTPError):
            c.cluster_delete(cluster['id'])

        # Delete node
        del_node = c.node_delete(node['id'])
        self.assertEqual(True, del_node)

        # Delete cluster
        del_cluster = c.cluster_delete(cluster['id'])
        self.assertEqual(True, del_cluster)

    def test_device(self):
        # Create app
        c = HeketiClient(TEST_SERVER, "admin", TEST_ADMIN_KEY)

        # Create cluster
        cluster = c.cluster_create()
        self.assertEqual(True, cluster['id'] != '')

        # Create node
        node_req = {}
        node_req['cluster'] = cluster['id']
        node_req['zone'] = 10
        node_req['hostnames'] = {
            "manage": ["node1-manage.gluster.lab.com"],
            "storage": ["node1-storage.gluster.lab.com"]
        }

        node = c.node_add(node_req)
        self.assertEqual(True, node['id'] != '')

        # Create a device request
        device_req = {}
        device_req['name'] = "sda"
        device_req['node'] = node['id']

        device = c.device_add(device_req)
        self.assertEqual(True, device)

        # Get node information
        info = c.node_info(node['id'])
        self.assertEqual(True, len(info['devices']) == 1)
        self.assertEqual(True, len(info['devices'][0]['bricks']) == 0)
        self.assertEqual(
            True, info['devices'][0]['name'] == device_req['name'])
        self.assertEqual(True, info['devices'][0]['id'] != '')

        # Get info from an unknown id
        with self.assertRaises(requests.exceptions.HTTPError):
            c.device_info("badid")

        # Get device information
        device_id = info['devices'][0]['id']
        device_info = c.device_info(device_id)
        self.assertEqual(True, device_info == info['devices'][0])

        # Set offline
        state = {}
        state['state'] = 'offline'
        self.assertEqual(True, c.device_state(device_id, state))

        # Get device info
        info = c.device_info(device_id)
        self.assertEqual(info['state'], 'offline')

        state['state'] = 'online'
        self.assertEqual(True, c.device_state(device_id, state))

        info = c.device_info(device_id)
        self.assertEqual(info['state'], 'online')

        # Try to delete node, and will not until we delete the device
        with self.assertRaises(requests.exceptions.HTTPError):
            c.node_delete(node['id'])

        # Delete unknown device
        with self.assertRaises(requests.exceptions.HTTPError):
            c.node_delete("badid")

        # Delete device
        device_delete = c.device_delete(device_info['id'])
        self.assertEqual(True, device_delete)

        # Delete node
        node_delete = c.node_delete(node['id'])
        self.assertEqual(True, node_delete)

        # Delete cluster
        cluster_delete = c.cluster_delete(cluster['id'])
        self.assertEqual(True, cluster_delete)

    def test_volume(self):
        # Create cluster
        c = HeketiClient(TEST_SERVER, "admin", TEST_ADMIN_KEY)
        self.assertEqual(True, c != '')

        cluster = c.cluster_create()
        self.assertEqual(True, cluster['id'] != '')

        # Create node request packet
        print "Creating Cluster"
        for i in range(3):
            node_req = {}
            node_req['cluster'] = cluster['id']
            node_req['hostnames'] = {
                "manage": ["node%s-manage.gluster.lab.com" % (i)],
                "storage": ["node%s-storage.gluster.lab.com" % (i)]}
            node_req['zone'] = i + 1

            # Create node
            node = c.node_add(node_req)
            self.assertEqual(True, node['id'] != '')

            # Create and add devices
            for i in range(1, 4):
                device_req = {}
                device_req['name'] = "sda%s" % (i)
                device_req['node'] = node['id']

                device = c.device_add(device_req)
                self.assertEqual(True, device)

        # Get list of volumes
        list = c.volume_list()
        self.assertEqual(True, len(list['volumes']) == 0)

        # Create a volume
        print "Creating a volume"
        volume_req = {}
        volume_req['size'] = 10
        volume = c.volume_create(volume_req)
        self.assertEqual(True, volume['id'] != "")
        self.assertEqual(True, volume['size'] == volume_req['size'])

        # Get list of volumes
        list = c.volume_list()
        self.assertEqual(True, len(list['volumes']) == 1)
        self.assertEqual(True, list['volumes'][0] == volume['id'])

        # Get info on incorrect id
        with self.assertRaises(requests.exceptions.HTTPError):
            c.volume_info("badid")

        # Get info
        info = c.volume_info(volume['id'])
        self.assertEqual(True, info == volume)

        # Expand volume with a bad id
        volume_ex_params = {}
        volume_ex_params['expand_size'] = 10

        with self.assertRaises(requests.exceptions.HTTPError):
            c.volume_expand("badid", volume_ex_params)

        # Expand volume
        print "Expanding volume"
        volumeInfo = c.volume_expand(volume['id'], volume_ex_params)
        self.assertEqual(True, volumeInfo['size'] == 20)

        # Delete bad id
        with self.assertRaises(requests.exceptions.HTTPError):
            c.volume_delete("badid")

        # Delete volume
        print "Deleting volume"
        volume_delete = c.volume_delete(volume['id'])
        self.assertEqual(True, volume_delete)

        print "Deleting Cluster"
        clusterInfo = c.cluster_info(cluster['id'])
        for node_id in clusterInfo['nodes']:
            # Get node information
            nodeInfo = c.node_info(node_id)

            # Delete all devices
            for device in nodeInfo['devices']:
                device_delete = c.device_delete(device['id'])
                self.assertEqual(True, device_delete)

            # Delete node
            node_delete = c.node_delete(node_id)
            self.assertEqual(True, node_delete)

        # Delete cluster
        cluster_delete = c.cluster_delete(cluster['id'])
        self.assertEqual(True, cluster_delete)


if __name__ == '__main__':
    unittest.main()
