variable "foo" {
	default = "bar"
	description = "bar"
}

variable "groups" { }

provider "aws" {
	access_key = "foo"
	secret_key = "bar"
}

provider "do" {
	api_key = "${var.foo}"
}

resource "aws_security_group" "firewall" {
	count = 5
}

resource aws_instance "web" {
	ami = "${var.foo}"
	security_groups = [
		"foo",
		"${aws_security_group.firewall.foo}",
		"${element(split(\",\", var.groups)}",
	]
	network_interface = {
		device_index = 0
		description = "Main network interface"
	}
}

resource "aws_instance" "db" {
	security_groups = "${aws_security_group.firewall.*.id}"
	VPC = "foo"
	depends_on = ["aws_instance.web"]
}

output "web_ip" {
	value = "${aws_instance.web.private_ip}"
}
