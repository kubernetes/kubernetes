# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'prometheus/client/model/version'

Gem::Specification.new do |spec|
  spec.name          = 'prometheus-client-model'
  spec.version       = Prometheus::Client::Model::VERSION
  spec.authors       = ['Tobias Schmidt']
  spec.email         = ['tobidt@gmail.com']
  spec.summary       = 'Data model artifacts for the Prometheus Ruby client'
  spec.homepage      = 'https://github.com/prometheus/client_model/tree/master/ruby'
  spec.license       = 'Apache 2.0'

  spec.files         = %w[README.md LICENSE] + Dir.glob('{lib/**/*}')
  spec.require_paths = ['lib']

  spec.add_dependency 'beefcake', '>= 0.4.0'

  spec.add_development_dependency 'bundler', '~> 1.3'
  spec.add_development_dependency 'rake'
end
