# Ansible
## Ansible introduction
Ansible is an open-source software provisioning, configuration management, and application-deployment tool enabling infrastructure as code. It runs on many Unix-like systems, and can configure both Unix-like systems as well as Microsoft Windows. It includes its own declarative language to describe system configuration.
We will use Ansible as a instrumentation engine. It will run infrastructure deployments, Kubeflow and additional software stack installation. It's yaml based syntax will serve also as a great human-redable documentation.

Ansible helps to manage multiple machines by selecting portions of Ansible's inventory stored in simple ASCII text files. The inventory is configurable, and target machine inventory can be sourced dynamically or from cloud-based sources in different formats (YAML, INI).[13]

Sensitive data can be stored in encrypted files using Ansible Vault. In contrast with other popular configuration-management software — such as Chef, Puppet, and CFEngine — Ansible uses an agentless architecture,[16] with Ansible software not normally running or even installed on the controlled node.[16] Instead, Ansible orchestrates a node by installing and running modules on the node temporarily via SSH. For the duration of an orchestration task, a process running the module communicates with the controlling machine with a JSON-based protocol via its standard input and output.[17] When Ansible is not managing a node, it does not consume resources on the node because no daemons are run or software installed.
## Ansible Design
The design goals of Ansible include:


    * Minimal in nature. Management systems should not impose additional dependencies on the environment.
    * Consistent. With Ansible one should be able to create consistent environments.
    * Secure. Ansible does not deploy agents to nodes. Only OpenSSH and Python are required on the managed nodes.
    * Reliable. When carefully written, an Ansible playbook can be idempotent, to prevent unexpected side-effects on the managed systems. It is possible to write playbooks that are not idempotent.
    * Minimal learning required. Playbooks use an easy and descriptive language based on YAML and Jinja templates.
## Ansible Modules
Modules are mostly standalone and can be written in a standard scripting language (such as Python, Perl, Ruby, Bash, etc.). One of the guiding properties of modules is idempotency, which means that even if an operation is repeated multiple times (e.g., upon recovery from an outage), it will always place the system into the same state

## Ansible Inventory
Location of target nodes is specified through inventory configuration lists (INI or YAML formatted) .

## Ansible Playbooks
Playbooks are YAML files that store lists of tasks for repeated executions on managed nodes. Each Playbook maps (associates) a group of hosts to a set of roles. Each role is represented by calls to Ansible tasks.

# Ansible structure for Kubeflow installation

## Design directory

1 Create 'ansible' directory. It will be our main directory.

2 In 'ansible' dir create 'group_vars', 'inventory' and 'roles' dirs

3 create file 'ansible/group_vars/demo' with content:
```
demo_var: "hello"
```
4 create ansible/inventory/local
```
[local]
localhost ansible_connection=local ansible_python_interpreter=/path/to/venv/bin/python
```
5 create ansible/roles/demo/tasks/main.yaml
```
- name: demo
  shell: "echo {{ demo_var }}"
  args:
    executable: /bin/bash
```
6 create ansible/ansible.cfg
```
[defaults]
bin_ansible_callbacks =True
display_skipped_hosts = no
host_key_checking = False
stdout_callback = yaml


[inventory]
enable_plugins = host_list, script, yaml, ini, auto
```
7 create ansible/noble_demo.yaml
```
- hosts: local

  vars_files:
  - group_vars/demo

  roles:
  - demo
```
You should have the following structure:
```
.
├── Makefile
├── ansible
│   ├── ansible.cfg
│   ├── group_vars
│   │   ├── demo
│   ├── inventory
│   │   └── local
│   ├── noble_demo.yaml
│   └── roles
│   │  ├── demo
│   │  │   ├── tasks
│   │  │   ├── ├──  main.yaml
```
```

8 Run it with(you can add it to Makefile):
```
cd ansible && ansible-playbook -i inventory/local noble_demo.yaml --verbose
```
9 Get familiar with ansible modules:
https://docs.ansible.com/ansible/2.7/modules/list_of_all_modules.html 

10 Try to add something to the demo playbook...
