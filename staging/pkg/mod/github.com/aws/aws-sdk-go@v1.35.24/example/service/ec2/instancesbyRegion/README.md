
# Example  Fetch By region

This is an example using the AWS SDK for Go to list ec2 instances instance state By different region . By default it fetch all running and stopped instance


# Usage


```sh
# To fetch the stopped and running instances of all region use below:
./filter_ec2_by_region --state running --state stopped

# To fetch the stopped and running instances for region us-west-1 and eu-west-1 use below:
./filter_ec2_by_region --state running --state stopped --region us-west-1 --region=eu-west-1
```

## Sample Output

```
Fetching instace details  for region: ap-south-1 with criteria:  [running][stopped]**
 printing instance details.....
instance id i-************
current State stopped
done for region ap-south-1 ****



Fetching instace details  for region: eu-west-2 with criteria:  [running][stopped]**
 There is no instance for the for region eu-west-2 with the matching Criteria: [running][stopped]
done for region eu-west-2 ****
```
