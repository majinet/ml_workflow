# About this project

The objective of this project is to create a Cloud Native Computing Foundation (CNCF) standard Containeralize Platform for Data Scientists, I aspire to create a cloud-agnostic, decoupled and inter-dependent platform, seamlessly integrating AI to deliver complete automation and minimize operational costs, while reducing human error. As I look ahead, I firmly believe that the integration of DevOps, Containeralize Platform, and AI will revolutionize the tech industry. Leveraging these cutting-edge technologies, we can unlock immense productivity gains, streamline processes, and minimize downtime, while enhancing the overall customer experience.

# Machine Learning Devops (MLOps)

Machine Learning DevOps is a set of practices and tools for deploying, testing, and managing machine learning models in production environments. It combines the principles of DevOps with machine learning workflows, and aims to increase the speed and reliability of deploying and maintaining machine learning models.

The process of Machine Learning DevOps starts with the development and training of machine learning models, followed by packaging the models into deployable artifacts, such as Docker images. The artifacts are then deployed to the target environment, such as a Kubernetes cluster or a serverless platform. Continuous integration and continuous deployment (CI/CD) pipelines are often used to automate the process of building, testing, and deploying machine learning models.

Machine Learning DevOps also involves monitoring and managing the performance of the deployed models, as well as updating and retraining the models when new data is available or when the models no longer perform adequately. This involves creating feedback loops between the deployed models and the data sources, and incorporating automated testing, monitoring, and logging into the deployment pipelines.

Overall, Machine Learning DevOps is a critical practice for organizations looking to deploy and scale machine learning models in production environments, while ensuring that the models are reliable, efficient, and continuously improving.

## MLOps Level 4

MLOps Level 4 is a stage of maturity in the field of Machine Learning Operations (MLOps). At this level, organizations have achieved a high degree of automation and integration in their machine learning workflows, enabling continuous integration, delivery, and deployment of machine learning models at scale.

At MLOps level 4, organizations have established a fully automated end-to-end pipeline for machine learning models, from data ingestion and preprocessing to model training, evaluation, and deployment. This includes automated testing and monitoring of models in production, with feedback loops to continuously improve model performance.

MLOps level 4 also involves the use of advanced tools and techniques for managing and scaling machine learning operations, such as containerization, microservices, and cloud-based infrastructure. This allows organizations to manage large-scale machine learning workloads efficiently and cost-effectively.

Finally, at MLOps level 4, organizations have established a culture of collaboration and continuous improvement, with cross-functional teams working together to refine and optimize machine learning workflows and processes.

Adopting MLops level 4 can provide several benefits, including:

Faster time-to-market: By automating the deployment, testing, and monitoring of machine learning models, MLops level 4 can help organizations reduce the time it takes to bring new models into production.

Improved accuracy and performance: MLops level 4 can help organizations improve the accuracy and performance of their machine learning models by enabling continuous training and testing in a production-like environment.

Increased scalability: With MLops level 4, organizations can scale their machine learning operations more effectively by automating processes and using cloud-based infrastructure.

Reduced operational costs: By automating processes and reducing the need for manual intervention, MLops level 4 can help organizations reduce their operational costs and improve efficiency.

Improved collaboration: MLops level 4 can facilitate collaboration between data scientists, developers, and operations teams by providing a shared platform for managing machine learning models and workflows.

## Continuous Integration (CI)

Continuous Integration (CI) is the practice of regularly and automatically building, testing, and integrating code changes into a shared repository. In the context of Machine Learning, CI involves automating the build, test, and deployment processes of machine learning models. This ensures that changes to a model, its features, or its infrastructure can be easily integrated and tested before being released into production.

In a typical CI process for Machine Learning, the code changes are automatically checked out from the code repository, model training is performed, and the model is tested to ensure that it meets certain performance metrics. This process can also include validation of input data and feature engineering pipelines, as well as testing the model on different hardware and software environments.

The goal of continuous integration is to catch potential errors or issues early in the development process, reducing the amount of time and resources needed to fix them. By automating this process, it also helps to increase the reliability and consistency of machine learning models.

Some common tools used in CI for Machine Learning include Jenkins, Travis CI, CircleCI, GitLab CI/CD, and GitHub Actions. These tools provide an easy and automated way to perform the build, test, and deployment steps required for Machine Learning DevOps.


### Overall Architecture

![Alt text](screens/Overall_Arch.jpg "CI Platform")

Core Components:

- Kubernetes
- [Kubeflow Pipeline](pipelines)
- Feast (Feature Store)
- Docker & Docker Hub (Container Image Registry)
- Minio (Object Store)
- KATib (Hyperparameters Tuning)
- [GitHub & GitHub Actions](.github/workflows)


### Why Cloud Native Computing Foundation (CNCF) Compliance tools are important to MLOps

Cloud Native Computing Foundation (CNCF) compliance tools provide a set of best practices and guidelines for developing, deploying, and managing cloud-native applications. Here are some reasons to adopt these tools in MLOps:

Standardization: CNCF compliance tools help standardize the way MLOps teams develop, deploy, and manage machine learning models. This can lead to more consistent and reliable workflows, making it easier to maintain and scale the models.

Portability: By following the CNCF compliance guidelines, MLOps teams can ensure that their machine learning models can run on any cloud infrastructure that supports these guidelines. This can help avoid vendor lock-in and make it easier to migrate to different cloud platforms.

Security: The CNCF compliance tools provide guidelines for secure development, deployment, and management of cloud-native applications. By following these guidelines, MLOps teams can ensure that their machine learning models are secure and protected against potential threats.

Scalability: The CNCF compliance tools provide best practices for designing scalable and resilient cloud-native applications. By following these guidelines, MLOps teams can ensure that their machine learning models can handle high volumes of data and traffic without any performance issues.

Community Support: CNCF compliance tools are developed and maintained by a large community of experts and contributors. This means that MLOps teams can benefit from a wide range of resources, including documentation, tutorials, and support from the community.

### Why create a cloud-agnostic platform in MLOps

There are several benefits of creating a cloud-agnostic platform, including:

Flexibility and freedom: A cloud-agnostic platform allows businesses to choose the cloud provider that best meets their needs, without being locked into a specific vendor or technology. This provides greater flexibility and freedom to adapt to changing business requirements.

Cost savings: By leveraging multiple cloud providers, businesses can negotiate better pricing and take advantage of cost savings opportunities, such as spot instances or reserved instances.

Improved scalability: A cloud-agnostic platform can be scaled up or down quickly and easily, allowing businesses to respond to changes in demand without having to worry about infrastructure limitations or vendor lock-in.

Portability: With a cloud-agnostic platform, applications can be easily moved from one cloud provider to another without the need for extensive modifications or changes to the underlying architecture.

Reduced risk: By avoiding vendor lock-in and relying on multiple cloud providers, businesses can reduce their risk of downtime or service disruptions caused by outages or other issues with a single provider.

### Why use Open-source tools in MLOps

Using open-source tools in MLOps provides several advantages:

Cost-effective: Open-source tools are usually free to use and distribute, which can be particularly beneficial for smaller companies or teams with limited budgets.

Flexibility: Open-source tools are typically customizable and can be adapted to fit specific use cases or requirements.

Large community support: Open-source tools usually have a large community of developers and users, which can provide a wealth of knowledge and resources for troubleshooting and improving the tools.

Transparency: Open-source tools are open to inspection and modification by anyone, which promotes transparency and accountability.

Integration: Open-source tools can be integrated with other open-source tools, proprietary tools, or cloud services, which can help to create a more comprehensive and scalable MLOps platform.

Innovation: The open-source community is often at the forefront of innovation, which means that open-source tools are often the first to adopt new technologies or techniques.

Overall, using open-source tools in MLOps can help teams to create a more flexible, cost-effective, and innovative platform that is well-supported by a large community of developers and users.


### Why adopt Containerization infrastructure in Machine Learning System

Containerization infrastructure has several benefits when it comes to Machine Learning systems, including:

Portability: Containers provide a consistent runtime environment, making it easier to move a machine learning system from one environment to another, such as from a development environment to a production environment, or between different cloud providers.

Scalability: Containers are designed to be lightweight and easy to deploy, which makes it simple to scale up or down a machine learning system to meet changing demands.

Reproducibility: Containers can be versioned, allowing for reproducible builds and deployments. This ensures that the machine learning system behaves consistently across different environments and can be debugged more easily.

Isolation: Containers provide an isolated environment for running the machine learning system, preventing conflicts with other software running on the same machine.

Resource Efficiency: Containers allow for more efficient use of resources, such as CPU and memory, compared to running machine learning systems on virtual machines or dedicated hardware.

Overall, containerization infrastructure provides a more efficient and streamlined way to develop, test, and deploy machine learning systems, leading to faster time-to-market and increased agility.


### What is right size for DevOps team?

DevOps team members consist of :
- Cloud Engineers
- Network Engineers
- Developers (Java, Python)
- Operators (operate and monitor system)
- Security experts
- Database Administrators
- Quality Assurance Engineers
- DevOps Engineers / Automation experts
- Solution Architects
- Project Managers

## Continuous Deployment (CD)

### TBC
