kubectl -n kafka run kafka-producer -ti --image=quay.io/strimzi/kafka:0.35.0-kafka-3.4.0 --rm=true --restart=Never -- bin/kafka-console-producer.sh --bootstrap-server my-cluster-kafka-bootstrap:9092 --topic my-topic
kubectl -n kafka run kafka-consumer -ti --image=quay.io/strimzi/kafka:0.35.0-kafka-3.4.0 --rm=true --restart=Never -- bin/kafka-console-consumer.sh --bootstrap-server my-cluster-kafka-bootstrap:9092 --topic feast.public.titanic_survive_svc_v1 --from-beginning

# read key in message
kubectl -n kafka run kafka-consumer -ti --image=quay.io/strimzi/kafka:0.35.0-kafka-3.4.0 --rm=true --restart=Never -- bin/kafka-console-consumer.sh --bootstrap-server my-cluster-kafka-bootstrap:9092 --topic feast.public.titanic_survive_svc_v1 --property print.key=true --from-beginning

kubectl -n kafka run kafka-topics-list -ti --image=quay.io/strimzi/kafka:0.35.0-kafka-3.4.0 --rm=true --restart=Never -- bin/kafka-topics.sh --bootstrap-server my-cluster-kafka-bootstrap:9092 --list
kubectl -n kafka run kafka-topics-list -ti --image=quay.io/strimzi/kafka:0.35.0-kafka-3.4.0 --rm=true --restart=Never -- bin/kafka-topics.sh --bootstrap-server my-cluster-kafka-bootstrap:9092 --describe --topic feast.public.titanic_survive_svc_v1



