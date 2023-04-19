## Ubuntu升级

* ### [Fixing "The following packages have been kept back" Error While Updating Ubuntu and Debian-based Linux Distributions](https://itsfoss.com/following-packages-have-been-kept-back/)

> In the example here, it requires the removal of fwupd-signed package.

> This is a mechanism in Debian’s APT package manager that informs you that an installed package needs to remove dependency packages. When you manually and individually update these packages, the system understands that you are willing to remove the dependency package.

* ### [I'm getting the message: "The following security updates require Ubuntu Pro with 'esm-apps' enabled" when updating Ubuntu 22.04](https://askubuntu.com/questions/1452299/im-getting-the-message-the-following-security-updates-require-ubuntu-pro-with)


* ### [如何在线将Ubuntu 18.04升级到Ubuntu 20.04](https://blog.csdn.net/smartvxworks/article/details/119175947)

* ### [Ubuntu 18.04LTS升级到20.04版本 - TNEXT](https://tnext.org/6680.html)


<br>

## Ubuntu升级之后可能引发的docker的问题

* [Cannot connect to the Docker daemon at unix:/var/run/docker.sock.](https://stackoverflow.com/questions/44678725/cannot-connect-to-the-docker-daemon-at-unix-var-run-docker-sock-is-the-docker)

* [Failed to start Docker Application Container Engine](https://stackoverflow.com/questions/49110092/failed-to-start-docker-application-container-engine)
    ```
    sudo dockerd
    ```

* [failed to start daemon: Error initializing network controller: Error creating default "bridge" network](https://stackoverflow.com/questions/65213831/failed-to-start-daemon-error-initializing-network-controller-error-creating-de)

    ```
    $ sudo firewall-cmd --permanent --zone=docker --change-interface=docker0
    $ sudo firewall-cmd --reload
    ```

* [Job for docker.service failed because the control process exited with error code (Pi-hole)](https://forums.docker.com/t/job-for-docker-service-failed-because-the-control-process-exited-with-error-code-pi-hole/131461)

    ```
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    ```

* [如何在Ubuntu 20.04安装Docker](https://www.myfreax.com/how-to-install-and-use-docker-on-ubuntu-20-04/)

    ```
    sudo apt update 
    sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common -y 

    sudo apt-get remove docker docker.io containerd runc -y 

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 

    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" 

    sudo apt update 
    sudo apt install docker-ce docker-ce-cli containerd.io -y
    ```

* [docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]](https://askubuntu.com/questions/1400476/docker-error-response-from-daemon-could-not-select-device-driver-with-capab)
  * 具体安装步骤可参见：[Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/archive/1.8.1/install-guide.html#setting-up-nvidia-container-toolkit)
  * 注意重启docker服务这步必不可少
    ```
    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    $ sudo apt-get update

    $ sudo apt-get install -y nvidia-docker2

    $ sudo systemctl restart docker

    $ docker run --shm-size 2GB -it --gpus all docurdt/heal (base) root@9f66ed7b7c1b:/Workspace# 
    ```

<br>

## 显卡驱动更新

docker启动时可能会报错，这会涉及到驱动更新，可参照如下操作：

* [Linux Troubleshooting](https://docs.omniverse.nvidia.com/prod_launcher/prod_kit/linux-troubleshooting.html)

    ```
    sudo nvidia-uninstall
    sudo apt-get remove --purge nvidia-*
    sudo apt autoremove
    sudo apt autoclean
    ```
    * 如果有图形界面，需额外运行：
    ```
    sudo systemctl isolate multi-user.target
    ```
* [Container Installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_container.html#container-setup)

    ```
    $ sudo apt-get update
    $ sudo apt install build-essential -y
    $ wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.11/NVIDIA-Linux-x86_64-525.60.11.run
    $ chmod +x NVIDIA-Linux-x86_64-525.60.11.run
    $ sudo ./NVIDIA-Linux-x86_64-525.60.11.run
    ```

<br>

