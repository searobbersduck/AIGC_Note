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

