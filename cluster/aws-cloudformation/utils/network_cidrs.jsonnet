/*
 * Common network map shared across regions and AZs; includes utility accessor functions.
 */

{
  /**
   * Map of AWS region to values for its dedicated VPC address space and overlay network address space.
   */
  local regions = {
    "us-west-2": { VPC: "172.16.0.0/16", Overlay: "10.0.0.0/14" },
    "us-west-1": { VPC: "172.17.0.0/16", Overlay: "10.4.0.0/14" },
    "us-east-1": { VPC: "172.18.0.0/16", Overlay: "10.8.0.0/14" },
    "eu-west-1": { VPC: "172.19.0.0/16", Overlay: "10.12.0.0/14" },
    "eu-central-1": { VPC: "172.20.0.0/16", Overlay: "10.16.0.0/14" },
    "ap-southeast-1": { VPC: "172.21.0.0/16", Overlay: "10.20.0.0/14" },
    "ap-southeast-2": { VPC: "172.22.0.0/16", Overlay: "10.24.0.0/14" },
    "ap-northeast-1": { VPC: "172.23.0.0/16", Overlay: "10.28.0.0/14" },
    "sa-east-1": { VPC: "172.24.0.0/16", Overlay: "10.32.0.0/14" },
    corp: { VPC: "172.25.0.0/16", Overlay: "" },  // Reserve IP space for corp offices.
    dev: { VPC: "172.26.0.0/16", Overlay: "10.248.0.0/16" },  // Reserve IP space for local dev environments
  },

  /**
   * Map splitting each active region's VPC space into public and private subnet CIDRs.
   *
   * Typically, split each regional /16 into 4 /18s, one for a private subnet for each of three AZs in the region
   * hosting Kubernetes and the remaining one split into /20s for use in corresponding public subnets.
   */
  local zones = {
    "us-west-2a": { Public: "172.16.0.0/20", Private: "172.16.64.0/18" },
    "us-west-2b": { Public: "172.16.16.0/20", Private: "172.16.128.0/18" },
    "us-west-2c": { Public: "172.16.32.0/20", Private: "172.16.192.0/18" },
  },

  /**
   * IP Space allocated to corp offices, if any.
   *
   * Keeping these from overlapping with the VPC space above will allow for a smooth transition to a VPN connection
   * between sites and the VPC itself.
   */
  local corp_offices = {
    //"SFO": { "PublicCidr": "0.0.0.0/32", "PrivateCidr": "172.25.0.0/20"},
    //"SEA": { "PublicCidr": "0.0.0.0/32", "PrivateCidr": "172.25.16.0/20"},
  },

  /**
   * Non-routable Kubernetes "service" blocks - these can overlap across clusters as they're not routable,
   * but should not overlap with any routable addresses.
   */
  local kube_service_block = "10.240.0.0/14",

  // Retrieves the CIDR block for a given region's VPC.
  vpc_cidr_block(region)::
    regions[region].VPC,

  // Retrieves the CIDR block for a given region's Overlay network.
  overlay_cidr_block(region)::
    regions[region].Overlay,

  // Retrieves the CIDR block for a given region's Kubernetes services.
  kube_services_cidr_block(region)::
    kube_service_block,

  // Retrieves the IP for a given region's Kubernetes internal DNS service endpoint.
  kube_internal_dns_ip(region)::
    "10.240.0.10",

  // Retreives the CIDR block for the public subnet in the given zone.
  public_subnet_cidr_block(zone)::
    zones[zone].Public,

  // Retreives the CIDR block for the private subnet in the given zone.
  private_subnet_cidr_block(zone)::
    zones[zone].Private,

  // Retrieves the list of registered offices.
  corp_offices_list()::
    std.objectFields(corp_offices),

  // Retrieves the list of external corp CIDR blocks.
  corp_cidr_blocks_list()::
    [corp_offices[x].PublicCidr for x in std.objectFields(corp_offices)],

  // Retrieves the list of internal corp CIDR blocks.
  corp_internal_cidr_blocks_list()::
    [corp_offices[x].PrivateCidr for x in std.objectFields(corp_offices)],

  corp_office_public_ip(office_tla)::
    std.split(corp_offices[office_tla].PublicCidr, "/")[0],
}
