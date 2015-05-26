app.provider('k8sApi',
             function() {

               var urlBase = '';

               this.setUrlBase = function(value) { urlBase = value; };

               var _get = function($http, baseUrl, query) {
                 var _fullUrl = baseUrl;
                 if (query !== undefined) {
                   _fullUrl += '/' + query;
                 }

                 return $http.get(_fullUrl);
               };

               this.$get = function($http, $q) {
                 var api = {};

                 api.getUrlBase = function() { return urlBase; };

                 api.getPods = function(query) { return _get($http, urlBase + '/pods', query); };

                 api.getMinions = function(query) { return _get($http, urlBase + '/nodes', query); };

                 api.getNodes = api.getMinions;

                 api.getServices = function(query) { return _get($http, urlBase + '/services', query); };

                 api.getReplicationControllers = function(query) {
                   return _get($http, urlBase + '/replicationControllers', query)
                 };

                 api.getEvents = function(query) { return _get($http, urlBase + '/events', query); };

                 return api;
               };
             })
    .config(function(k8sApiProvider, ENV) {
      if (ENV && ENV['/'] && ENV['/']['k8sApiServer']) {
        var proxy = ENV['/']['cAdvisorProxy'] || '';
        k8sApiProvider.setUrlBase(proxy + ENV['/']['k8sApiServer']);
      }
    });

app.provider('k8sv1Beta3Api',
             function() {

               var urlBase = '';
               var _namespace = 'default';

               this.setUrlBase = function(value) { urlBase = value; };

               this.setNamespace = function(value) { _namespace = value; };
               this.getNamespace = function() { return _namespace; };

               var _get = function($http, baseUrl, query) {
                 var _fullUrl = baseUrl;

                 if (query !== undefined) {
                   _fullUrl += '/' + query;
                 }

                 return $http.get(_fullUrl);
               };

               this.$get = function($http, $q) {
                 var api = {};

                 api.getUrlBase = function() { return urlBase + '/namespaces/' + _namespace; };

                 api.getPods = function(query) { return _get($http, api.getUrlBase() + '/pods', query); };

                 api.getMinions = function(query) { return _get($http, urlBase + '/nodes', query); };

                 api.getServices = function(query) { return _get($http, api.getUrlBase() + '/services', query); };

                 api.getReplicationControllers = function(query) {
                   return _get($http, api.getUrlBase() + '/replicationcontrollers', query)
                 };

                 api.getEvents = function(query) { return _get($http, api.getUrlBase() + '/events', query); };

                 return api;
               };
             })
    .config(function(k8sv1Beta3ApiProvider, ENV) {
      if (ENV && ENV['/'] && ENV['/']['k8sApiv1beta3Server']) {
        var proxy = ENV['/']['cAdvisorProxy'] || '';
        k8sv1Beta3ApiProvider.setUrlBase(proxy + ENV['/']['k8sApiv1beta3Server']);
      }
    });
