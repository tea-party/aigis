# aigis

aigis is an ai chatbot, made in rust, running on the bluesky network. her main job is to learn from the network by chatting with users, understand them a little, and keep what matters in memory. think of her as a tool designed to fight the chaos of too much information, like a weapon, but for clarity and insight. of course, some say she was made for something more. something older. something meant to stand against the unseen, the quiet disruptions beneath the surface. something more than just data.


## what she does

aigis connects to bluesky’s live data feed, called the `jetstream` and reads new posts and other events as they happen. using `fastembed`, aigis turns her chats with others into embeddings, a numerical way of understanding relationships. these get saved into a qdrant vector database which acts as her long-term memory. the idea is that aigis can use this memory to understand context over time, eventually allowing her to respond intelligently or possibly even to take action on the network automatically.

## main parts

* **jetstream listener**: connects and listens to bluesky’s data stream with `rocketman`
* **text processor**: converts posts into embeddings using `fastembed`
* **memory core**: stores and retrieves embeddings in a qdrant vector database
* **agent logic** (in progress): will decide what to do based on new data and memory

## how to start

set your environment variables for your bluesky account (`atp_user`, `atp_password`) and your qdrant connection. then build and run the rust project. aigis will start listening and learning.
