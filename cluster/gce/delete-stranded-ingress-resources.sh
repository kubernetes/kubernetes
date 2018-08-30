#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A utility for cleaning up GCE networking resources created through GCE Ingress
# controller upon deletion of the cluster before deletion of the Ingress.
set -euo pipefail
IFS=$'\n\t'

log() {
	echo "$@" >&2
}

PROJECT="${PROJECT:?Required variable PROJECT.}"

gcloud() {
	command gcloud --project="${PROJECT}" "$@"
}

v_gcloud() {
	echo "+ gcloud --project=${PROJECT}" "$@"
	(
		set +e
		command gcloud --project="${PROJECT}" "$@"
	)
}

has_health_status() {
	local backend
	backend="$1"

	local out
	out="$(gcloud compute backend-services get-health "${backend}" \
		--global --format='value(status.healthStatus)')"
	[[ -n "$out" ]]
}

cleanup_resources() {
	local backend backend_id
	backend="$1"
	backend_id="$(echo "${backend}" | grep -Eo '\-\-([0-9a-f]+)' /dev/stdin | sed 's/^--//')"

	local health_check firewall url_map target_http_proxy target_https_proxy \
		forwarding_rule instance_groups

	firewall="k8s-fw-l7--${backend_id}"

	url_map="$(gcloud compute url-maps list --format='value(name)' \
		--filter=defaultService~"${backend}")"
	if [[ -n "${url_map}" ]]; then
		forwarding_rule="${url_map/-um-/-fw-}"
		target_http_proxy="$(gcloud compute target-http-proxies list	\
			--format='value(name)' --filter=urlMap~"${url_map}")"
		target_https_proxy="$(gcloud compute target-https-proxies list \
			--format='value(name)' --filter=urlMap~"${url_map}")"
	fi

	health_check="$(gcloud compute backend-services describe \
		--global "${backend}" --format='value(healthChecks)')"

	instance_groups=$(gcloud compute backend-services describe \
		--global "${backend}" --format='value(backends.group)' | tr \; '\n')

	log "    FIREWALL=${firewall}"
	log "    HEALTH_CHECK=${health_check}"
	log "    URL_MAP=${url_map}"
	log "    TARGET_HTTP_PROXY=${target_http_proxy}"
	log "    TARGET_HTTPS_PROXY=${target_https_proxy}"
	log "    FORWARDING_RULE=${forwarding_rule}"
	if [[ -n "${instance_groups}" ]]; then
		log "    INSTANCE_GROUPS:"
		for ig in ${instance_groups}; do
			log "    - ${ig}"
		done
	fi

	read -p "Clean these up? (y/N?) " -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		if [[ -n "${forwarding_rule}" ]]; then
			v_gcloud compute forwarding-rules delete --global "${forwarding_rule}" -q || true
		fi
		if [[ -n "${target_http_proxy}" ]]; then
			v_gcloud compute target-http-proxies delete "${target_http_proxy}" -q || true
		fi
		if [[ -n "${target_https_proxy}" ]]; then
			v_gcloud compute target-https-proxies delete "${target_https_proxy}" -q || true
		fi
		if [[ -n "${url_map}" ]]; then
			v_gcloud compute url-maps delete "${url_map}" -q || true
		fi
		v_gcloud compute backend-services delete --global "${backend}" -q || true
		v_gcloud compute firewall-rules delete "${firewall}" -q || true
		v_gcloud compute health-checks delete "${health_check}" -q || true
		if [[ -n "${instance_groups}" ]]; then
			for ig in ${instance_groups}; do
				v_gcloud compute instance-groups unmanaged delete -q "${ig}" || true
			done
		fi
	fi

}

main(){
	BACKEND_SERVICES=$(gcloud compute backend-services list --global \
		--format='value(name)' --filter='name ~ ^k8s-be-')
	log "Found $(echo -n "${BACKEND_SERVICES}" | wc -l) total global backends."
	for be in ${BACKEND_SERVICES}; do
		log "--> Backend: ${be}"
		if has_health_status "${be}"; then
			log "    Backend is still in use. (SKIP)"
		else
			log "    Has no health status (UNUSED)"
			cleanup_resources "${be}"
		fi
		log ""
	done
}

main
