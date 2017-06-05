/**
 * Maps of supported AWS EC2 AMIs.
 */
{
  // Map of AWS vanilla amazon linux HVM AMI ID by region
  local amazon_vanilla_linux_ami = {
    "us-west-2": "ami-63b25203",
  },

  // Map of AWS specialized VPC NAT instance HVM AMI ID by region
  // TODO: migrate to AWS NAT Gateway once available in CloudFormation.
  local amazon_vpc_nat_linux_ami = {
    "us-west-2": "ami-69ae8259",
  },

  // Map of CoreOS HVM AMI ID by region, pulled from https://coreos.com/os/docs/latest/booting-on-ec2.html
  local coreos_ami = {
    "eu-central-1": "ami-15190379",
    "ap-northeast-1": "ami-02c9c86c",
    "sa-east-1": "ami-c40784a8",
    "ap-southeast-2": "ami-949abdf7",
    "ap-southeast-1": "ami-00a06963",
    "us-east-1": "ami-7f3a0b15",
    "us-west-2": "ami-4f00e32f",
    "us-west-1": "ami-a8aedfc8",
    "eu-west-1": "ami-2a1fad59",
  },

  /**
   * Helper to retrieve the vanilla Amazon Linux AMI ID for the given region.
   */
  amazon_vanilla_ami_id(aws_region)::
    amazon_vanilla_linux_ami[aws_region],

  /**
   * Helper to retrieve the Amazon VPC NAT AMI ID for the given region.
   */
  amazon_vpc_nat_ami_id(aws_region)::
    amazon_vpc_nat_linux_ami[aws_region],

  /**
   * Helper to retrieve the CoreOS AMI ID for the given region.
   */
  coreos_ami_id(aws_region)::
    coreos_ami[aws_region],
}
