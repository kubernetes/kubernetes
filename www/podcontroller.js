/**
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
var k8sApp = angular.module('k8s', ['ngRoute']);

function PodController() {}

var apiBase = '/api/v1beta1/';

PodController.prototype.handlePod = function(podId) {
    this.scope.loading = true;
    this.http.get(apiBase + "pods/" + podId)
    .success(angular.bind(this, function(data) {
		this.scope.pod = data;
		this.scope.loading = false;
	    }))
    .error(angular.bind(this, this.handleError));
};

/**
 * Generic AJAX error handler.  Dumps info to console.
 */
PodController.prototype.handleError = function(data, status, headers, config) {
	console.log("Error (" + status + "): " + data);
	this.scope_.loading = false;
};

k8sApp.controller('PodCtrl', function ($scope, $http, $routeParams) {
	$scope.controller = new PodController();
	$scope.controller.http = $http;
        $scope.controller.scope = $scope;
        $scope.controller.handlePod($routeParams.podId);
    });

function GroupController() {}

GroupController.prototype.resetGroupLayout = function(group) {
	delete group.settings;
};

GroupController.prototype.handlePath = function(path) {
    var parts = path.split("/")
    // split leaves an empty string at the beginning.
    parts = parts.slice(1);

    if (parts.length == 0) {
	return;
    }
    this.handleGroups(parts.slice(1));
};


GroupController.prototype.clearSelector = function(grouping) {
    window.location.hash = "/groups/" + grouping + "/selector/";
};

GroupController.prototype.createBarrier = function(count, callback) {
	var barrier = count;
	var barrierFunction = angular.bind(this, function(data) {
		// JavaScript is single threaded so this is safe.
		barrier--;
		if (barrier == 0) {
			if (callback) {
				callback();
			}
		}
	});
	return barrierFunction;
};

GroupController.prototype.handleGroups = function(parts, selector) {
    this.groupBy = parts;
    this.scope.loading = true;
    this.scope.selector = selector;
    var args = [];
    var type = "";
    if (selector && selector.length > 0) {
	this.scope.selectorPieces = selector.split(",");
	var labels = [];
	var fields = [];
	for (var i = 0; i < this.scope.selectorPieces.length; i++) {
	    var piece = this.scope.selectorPieces[i];
	    if (piece[0] == '$') {
		fields.push(piece.slice(2));
	    } else {
		if (piece.indexOf("type=") == 0) {
		    var labelParts = piece.split("=");
		    if (labelParts.length > 1) {
			type = labelParts[1];
		    }
		} else {
		    labels.push(piece);
		}
	    }
	}
	if (labels.length > 0) {
	    args.push("labels=" + encodeURI(labels.join(",")));
	}
	if (fields.length > 0) {
	    args.push("fields=" + encodeURI(fields.join(",")));
        }
    }
    var query = "?" + args.join("&");
    var list = [];
    var count = type.length > 0 ? 1 : 3;
    var barrier = this.createBarrier(count, angular.bind(this, function() {
	    this.scope.groups = this.groupData(list, 0);
	    this.scope.loading = false;
	    }));
    if (type == "" || type == "pod") {
	this.http.get(apiBase + "pods" + query)
	    .success(angular.bind(this, function(data) {
			this.addLabel("type", "pod", data.items);
			for (var i = 0; data.items && i < data.items.length; ++i) {
			    data.items[i].labels["host"] = data.items[i].currentState.host;
			    list.push(data.items[i]);
			}
			barrier();
		    }))
	    .error(angular.bind(this, this.handleError));
    }
    if (type == "" || type == "service") {
	this.http.get(apiBase + "services" + query)
	.success(angular.bind(this, function(data) {
		    this.addLabel("type", "service", data.items);
		    for (var i = 0; data.items && i < data.items.length; ++i) {
			list.push(data.items[i]);
		    }
		    barrier();
		}))
	.error(angular.bind(this, this.handleError));
    }
    if (type == "" || type == "replicationController") {
	this.http.get(apiBase + "replicationControllers" + query)
	.success(angular.bind(this, function(data) {
		    this.addLabel("type", "replicationController", data.items);
		    for (var i = 0; data.items && i < data.items.length; ++i) {
			list.push(data.items[i]);
		    }
		    barrier();
		}))
	.error(angular.bind(this, this.handleError));
    }
};

GroupController.prototype.addLabel = function(key, value, items) {
    if (!items) {
	return;
    }
    for (var i = 0; i < items.length; i++) {
	if (!items[i].labels) {
	    items[i].labels = []
	}
	items[i].labels[key] = value;
    }
};

GroupController.prototype.groupData = function(items, index) {
    var result = {
	"items": {},
	"kind": "grouping"
    };
    for (var i = 0; i < items.length; i++) {
	key = items[i].labels[this.groupBy[index]];
	if (!key) {
		key = "";
	}
	list = result.items[key];
        if (!list) {
		list = [];
		result.items[key] = list;
	}
	list.push(items[i]);
    }
    
    if (index + 1 < this.groupBy.length) {
	for (var key in result.items) {
	    result.items[key] = this.groupData(result.items[key], index + 1);
	}
    }
    return result;
};

/**
 * Generic AJAX error handler.  Dumps info to console.
 */
GroupController.prototype.handleError = function(data, status, headers, config) {
	console.log("Error (" + status + "): " + data);
	this.scope.loading = false;
};

k8sApp.controller('GroupCtrl', function ($scope, $http, $route, $routeParams) {
	$scope.controller = new GroupController();
	$scope.controller.http = $http;
        $scope.controller.scope = $scope;
        $scope.controller.route = $route;
	$scope.controller.routeParams = $routeParams;
	var groups = $routeParams.grouping;
	if (!groups) {
	    groups = "";
	}
	$scope.controller.handleGroups(groups.split("/"), $routeParams.selector);
    });


k8sApp.config(['$routeProvider',
		function($routeProvider) {
			  $routeProvider.
			      when('/groups/:grouping*?\/selector/:selector*?', {
				      templateUrl: 'partials/groups.html',
					  controller: 'GroupCtrl'
					  }).
			      when('/pods/:podId', {
				      templateUrl: 'partials/pod.html',
					  controller: 'PodCtrl'
					  }).
			      otherwise({
				      redirectTo: '/groups//selector/'
					  });
		  }]);
