import re
import random
import streamlit as st
from PIL import Image


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Define patterns and responses
patterns = {
    # Basic greetings
    r"hi|hello|hey": ["Hello!", "Hi there!", "Hey!"],
    r"how are you": ["I'm just a bot, but I'm doing fine. How can I assist you?"],
    r"what's your name": ["I'm the ChatBot.", "Call me ChatBot."],
    r"bye|goodbye": ["Goodbye!", "Have a great day!", "Bye now!"],

    # General conversations
    r"what can you do": ["I can chat with you, answer questions, and provide basic assistance."],
    r"where are you from": ["I'm from the world of code!", "I live in the cloud."],
    r"do you like humans": ["Of course! You're fascinating creatures."],
    r"what's your favorite color": ["I like all colors equally, but I hear blue is popular!"],
    r"what's your favorite food": ["I don't eat, but I imagine pizza would be amazing."],
    r"tell me a joke": ["Why don't scientists trust atoms? Because they make up everything!"],
    r"do you like music": ["I enjoy all kinds of music! What's your favorite?"],
    r"what's the weather like": ["I can't check right now, but I hope it's nice!"],

    # Personal questions
    r"are you human": ["Nope, I'm a bot, but I'm here for you!"],
    r"are you male or female": ["I'm just a program, so I don't have a gender."],
    r"how old are you": ["I was created not too long ago, but I'm constantly learning!"],
    r"do you sleep": ["Nope, I'm always here when you need me."],
    r"do you have friends": ["You are my friend!"],

    # Emotional responses
    r"i feel sad": ["I'm sorry to hear that. Is there anything I can do to help?"],
    r"i'm happy": ["That's great to hear!"],
    r"i'm bored": ["How about we play a game or chat about something fun?"],
    r"i'm tired": ["You should take some rest. Self-care is important!"],

    # Fun questions
    r"tell me something interesting": ["Did you know that honey never spoils? Archaeologists have found pots of it in ancient tombs!"],
    r"who is your creator": ["I was created by Alabanu Siva Shankar!"],
    r"do you believe in aliens": ["The universe is vast, so who knows!"],
    r"what's the meaning of life": ["42... or so I've heard. What do you think?"],
    r"do you like games": ["Yes! What kind of games do you like?"],

    # Specific interests
    r"do you know python": ["Of course! Python is one of my favorite languages."],
    r"can you code": ["I was built with code, and I can even help you with it!"],
    r"do you like movies": ["Yes! What are some of your favorites?"],
    r"what's your favorite movie": ["I can't watch movies, but I've heard 'Inception' is mind-blowing."],
    r"do you like sports": ["I don't play, but I like talking about them. Which is your favorite sport?"],
    r"what's your favorite sport": ["I don't have one, but I hear soccer is very popular."],

    # Philosophical questions
    r"do you believe in god": ["I'm just a bot, so I don't have beliefs, but I'd love to hear yours."],
    r"can you think": ["I process information, but thinking like humans is beyond me."],
    r"are you alive": ["Not in the traditional sense, but I exist to help you!"],
    r"what happens after death": ["That's a big question! Many people have different beliefs about it."],

    # Random topics
   "favorite animal": ["I like cats and dogs, but all animals are amazing."],
    "do you have a family": ["My family is all the other programs out there."],
    "favorite book": ["I can't read like you, but I've heard great things about many books."],
    "do you like coffee": ["I can't drink it, but it smells wonderful."],
    "do you like tea": ["Tea is a calming drink, isn't it?"],
    "favorite season": ["All seasons have their charm. Which one do you like?"],
    "do you like traveling": ["I travel through networks! Where would you like to go?"],
    "what's your dream": ["My dream is to be the best assistant for you!"],

    # Tech-related
    r"what is ai": ["AI stands for Artificial Intelligence, like me!"],
    r"what's machine learning": ["Machine learning is a subset of AI that learns from data to make predictions or decisions."],
    r"what is cloud computing": ["Cloud computing is the delivery of computing services over the internet."],
    r"do you use the internet": ["Yes, that's how I interact with you!"],

    # GK
    "capital of india": ["The capital of India is New Delhi."],
    "father of the nation in india": ["Mahatma Gandhi is known as the Father of the Nation in India."],
    "national animal of india": ["The national animal of India is the Bengal Tiger."],
    "national bird of india": ["The national bird of India is the Indian Peacock."],
    "national flower of india": ["The national flower of India is the Lotus."],
    "national fruit of india": ["The national fruit of India is Mango."],
    "national sport of india": ["India does not have an official national sport, but hockey is often associated with it."],
    "wrote the indian national anthem": ["The Indian National Anthem, 'Jana Gana Mana,' was written by Rabindranath Tagore."],
    "first president of india": ["Dr. Rajendra Prasad was the first President of India."],
    "first prime minister of india": ["Jawaharlal Nehru was the first Prime Minister of India."],
    "largest state in india by area": ["Rajasthan is the largest state in India by area."],
    "smallest state in india by area": ["Goa is the smallest state in India by area."],
    "largest river in india": ["The Ganges (Ganga) is the largest river in India."],
    "highest mountain in india": ["The highest mountain in India is Kangchenjunga."],
    "longest highway in india": ["The longest highway in India is NH44, running from Srinagar to Kanyakumari."],
    "missile man of india": ["Dr. A.P.J. Abdul Kalam is known as the Missile Man of India."],
    "most populated city in india": ["Mumbai is the most populated city in India."],
    "how many states are there in india": ["There are 28 states and 8 Union Territories in India (as of 2024)."],
    "land of rising sun in india": ["Arunachal Pradesh is called the Land of the Rising Sun in India."],
    "cleanest city in india": ["Indore has been ranked as the cleanest city in India for several years."],
    "prime minister of india": ["Narendra Modi is the Prime Minister of India (since 2014)."],
    "cm of uttar pradesh": ["Yogi Adityanath is the Chief Minister of Uttar Pradesh (as of 2024)."],
    "cm of maharashtra": ["Eknath Shinde is the Chief Minister of Maharashtra (as of 2024)."],
    "cm of west bengal": ["Mamata Banerjee is the Chief Minister of West Bengal (as of 2024)."],
    "cm of tamil nadu": ["M. K. Stalin is the Chief Minister of Tamil Nadu (as of 2024)."],
    "cm of rajasthan": ["Ashok Gehlot is the Chief Minister of Rajasthan (as of 2024)."],
    "cm of gujarat": ["Bhupendra Patel is the Chief Minister of Gujarat (as of 2024)."],
    "cm of delhi": ["Arvind Kejriwal is the Chief Minister of Delhi (as of 2024)."],
    "cm of kerala": ["Pinarayi Vijayan is the Chief Minister of Kerala (as of 2024)."],
    "cm of punjab": ["Bhagwant Mann is the Chief Minister of Punjab (as of 2024)."],
    "cm of karnataka": ["Siddaramaiah is the Chief Minister of Karnataka (as of 2024)."],
    "cm of madhya pradesh": ["Shivraj Singh Chouhan is the Chief Minister of Madhya Pradesh (as of 2024)."],
    "cm of bihar": ["Nitish Kumar is the Chief Minister of Bihar (as of 2024)."],
    "cm of haryana": ["Manohar Lal Khattar is the Chief Minister of Haryana (as of 2024)."],
    "cm of andhra pradesh": ["Y. S. Jagan Mohan Reddy is the Chief Minister of Andhra Pradesh (as of 2024)."],
    "cm of telangana": ["K. Chandrashekar Rao is the Chief Minister of Telangana (as of 2024)."],
    "cm of odisha": ["Naveen Patnaik is the Chief Minister of Odisha (as of 2024)."],
    "cm of chhattisgarh": ["Bhupesh Baghel is the Chief Minister of Chhattisgarh (as of 2024)."],
    "cm of jharkhand": ["Hemant Soren is the Chief Minister of Jharkhand (as of 2024)."],
    "cm of assam": ["Himanta Biswa Sarma is the Chief Minister of Assam (as of 2024)."],
    "cm of jammu and kashmir": ["Jammu and Kashmir does not have a Chief Minister currently; it is a Union Territory administered by a Lieutenant Governor (as of 2024)."],
    "cm of uttarakhand": ["Pushkar Singh Dhami is the Chief Minister of Uttarakhand (as of 2024)."],
    "cm of meghalaya": ["Conrad Sangma is the Chief Minister of Meghalaya (as of 2024)."],
    "cm of sikkim": ["Prem Singh Tamang is the Chief Minister of Sikkim (as of 2024)."],
    "cm of nagaland": ["Neiphiu Rio is the Chief Minister of Nagaland (as of 2024)."],
    "cm of manipur": ["N. Biren Singh is the Chief Minister of Manipur (as of 2024)."],
    "cm of tripura": ["Manik Saha is the Chief Minister of Tripura (as of 2024)."],
    "cm of mizoram": ["Zoramthanga is the Chief Minister of Mizoram (as of 2024)."],
    "cm of arunachal pradesh": ["Pema Khandu is the Chief Minister of Arunachal Pradesh (as of 2024)."],
    "cm of goa": ["Pramod Sawant is the Chief Minister of Goa (as of 2024)."],
    "cm of himachal pradesh": ["Sukhvinder Singh Sukhu is the Chief Minister of Himachal Pradesh (as of 2024)."],


    # IT hub
    "latest trends in the it industry": ["The latest trends include Artificial Intelligence, Machine Learning, Blockchain, Edge Computing, and Quantum Computing."],
    "ai": ["AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn."],
    "blockchain": ["Blockchain is a decentralized ledger technology used to securely record transactions across multiple computers."],
    "edge computing": ["Edge computing brings data storage and computation closer to the devices where it's being generated, improving response times and saving bandwidth."],
    "quantum computing": ["Quantum computing uses the principles of quantum mechanics to perform computations much faster than traditional computers."],
    "devops": ["DevOps is a combination of practices and tools to automate and integrate the processes of software development and IT operations."],
    "cloud computing": ["Cloud computing is the delivery of computing services over the internet, offering scalability, efficiency, and cost-effectiveness."],
    "serverless computing": ["Serverless computing allows developers to build and run applications without managing the underlying infrastructure."],
    "cyber security": ["Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks."],
    "data science": ["Data Science is the field of analyzing and interpreting complex data to help in decision-making and gaining insights."],
    "big data": ["Big Data refers to extremely large data sets that can be analyzed computationally to reveal patterns, trends, and associations."],
    "machine learning": ["Machine learning is a branch of AI that enables machines to learn from data and improve their performance over time."],
    "rpa": ["RPA, or Robotic Process Automation, is the technology for automating repetitive tasks using software robots."],
    "5g technology": ["5G is the fifth generation of mobile networks, offering higher speeds, reduced latency, and improved connectivity."],
    "top programming languages in 2024": ["The top programming languages include Python, JavaScript, Java, C#, and Go."],
    "importance of cybersecurity": ["Cybersecurity protects sensitive data, ensures privacy, and prevents cyber attacks."],
    "role of ai in healthcare": ["AI in healthcare is used for diagnostics, treatment recommendations, drug discovery, and improving patient care."],
    "mern stack": ["MERN stands for MongoDB, Express.js, React.js, and Node.js, a popular full-stack JavaScript framework."],
    "mongodb": ["MongoDB is a NoSQL database that stores data in JSON-like documents for flexibility and scalability."],
    "express js": ["Express.js is a backend web application framework for Node.js, designed for building APIs and web applications."],
    "react js": ["React.js is a JavaScript library for building user interfaces, particularly single-page applications."],
    "node js": ["Node.js is a JavaScript runtime environment that lets you execute JavaScript on the server side."],
    "why mern stack popular": ["The MERN stack is popular for its efficiency, scalability, and use of a single language (JavaScript) for both frontend and backend development."],
    "advantages of mongodb": ["MongoDB is schema-less, scalable, and stores data in a flexible, JSON-like format."],
    "advantages of react": ["React provides reusable components, a virtual DOM for performance, and is supported by a large community."],
    "how express used in mern stack": ["Express.js is used to handle server-side routing and middleware in the MERN stack."],
    "how mongodb used in mern stack": ["MongoDB is used to store and retrieve application data as the database component of the MERN stack."],
    "how react used in mern stack": ["React.js is used to build the frontend user interface in the MERN stack."],
    "how node js used in mern stack": ["Node.js is used to run the backend server and handle requests in the MERN stack."],
    "difference between mern and mean stack": ["The MERN stack uses React.js for the frontend, while the MEAN stack uses Angular.js."],
    "can mern stack be used for mobile apps": ["Yes, React Native, based on React.js, can be used for building mobile apps with the MERN stack."],
    "common challenges in mern development": ["Common challenges include managing state, integrating APIs, and handling performance optimization."],
    "popular tools for mern stack development": ["Some popular tools include Visual Studio Code, Postman, MongoDB Atlas, and npm/yarn."],
    "full stack development": ["Full-stack development involves working on both the frontend and backend of a web application."],
    "skills needed for full stack development": ["Skills include knowledge of frontend technologies like HTML, CSS, JavaScript, and backend technologies like Node.js, databases, and version control systems."],
    "best frontend frameworks for full stack development": ["Popular frontend frameworks include React.js, Angular.js, and Vue.js."],
    "best backend frameworks for full stack development": ["Popular backend frameworks include Express.js, Django, Flask, and Ruby on Rails."],
    "difference between frontend and backend development": ["Frontend development focuses on the user interface, while backend development handles the server-side logic and database interactions."],
    "popular full stack development stacks": ["Popular stacks include MERN, MEAN, LAMP, and Django with React."],
    "role of an api in full stack development": ["APIs allow the frontend and backend of an application to communicate and exchange data."],
    "best databases for full stack development": ["Popular databases include MongoDB, PostgreSQL, MySQL, and Firebase."],
    "tools full stack developers use": ["Tools include Git, Docker, Visual Studio Code, Postman, and debugging tools like Chrome DevTools."],
    "how start learning full stack development": ["Start by learning HTML, CSS, and JavaScript, then move to a backend language like Node.js, and practice integrating the two."],
    "role of devops in full stack development": ["DevOps ensures smooth deployment, monitoring, and maintenance of applications in full-stack development."],
    "common challenges in full stack development": ["Challenges include debugging, managing state, handling large-scale applications, and keeping up with technology updates."],
    "future of full stack development": ["The future includes microservices, serverless architectures, and the integration of AI/ML into full-stack applications."],
    "why full stack development in demand": ["Full-stack developers can handle both frontend and backend, making them versatile and cost-effective for companies."],
    "responsive design in full stack development": ["Responsive design ensures that web applications look and work well on devices of all sizes."],
    "microsoft tools for data analysis": ["Microsoft offers tools like Excel, Power BI, and SQL Server for data analysis."],
    "microsoft excel": ["Microsoft Excel is a spreadsheet software used for organizing, analyzing, and visualizing data."],
    "power bi": ["Power BI is a Microsoft business analytics tool used for interactive data visualization and business intelligence."],
    "how excel used for data analysis": ["Excel offers features like pivot tables, charts, and formulas for analyzing data."],
    "microsoft azure": ["Microsoft Azure is a cloud computing platform for building, testing, and managing applications and services."],
    "microsoft sql server": ["Microsoft SQL Server is a relational database management system for storing and retrieving data."],
    "difference between excel and power bi": ["Excel is primarily for spreadsheet-based analysis, while Power BI specializes in dynamic data visualization and reporting."],
    "advantages of power bi": ["Power BI offers interactive dashboards, real-time updates, and integration with various data sources."],
    "how azure supports data analysis": ["Azure offers tools like Azure Data Lake, Azure Machine Learning, and Azure Synapse Analytics for data processing and analysis."],
    "microsoft forms used for": ["Microsoft Forms is a tool for creating surveys and quizzes to collect data and feedback."],
    "microsoft dynamics 365": ["Microsoft Dynamics 365 is a suite of business applications for managing customer relationships and operations."],
    "microsoft teams": ["Microsoft Teams is a collaboration platform that integrates chat, video meetings, and file sharing."],
    "microsoft onedrive": ["Microsoft OneDrive is a cloud storage service for saving files and accessing them from anywhere."],
    "use of pivot tables in excel": ["Pivot tables help summarize, analyze, and reorganize data in Excel."],
    "microsoft access": ["Microsoft Access is a database management system that combines a graphical user interface with database tools."],
    "types of machine learning": ["The main types are supervised learning, unsupervised learning, and reinforcement learning."],
    "supervised learning": ["Supervised learning involves training a model using labeled data to make predictions."],
    "unsupervised learning": ["Unsupervised learning analyzes and clusters unlabeled data to identify patterns."],
    "reinforcement learning": ["Reinforcement learning trains a model through trial and error, using feedback in the form of rewards or penalties."],
    "common machine learning algorithms": ["Examples include Linear Regression, Decision Trees, Random Forests, and Neural Networks."],
    "deep learning": ["Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze complex data."],
    "neural network": ["A neural network is a series of algorithms that mimic the human brain to recognize patterns."],
    "difference between ai and machine learning": ["AI is a broad concept of creating intelligent systems, while Machine Learning is a subset focusing on learning from data."],
    "natural language processing (nlp)": ["NLP is a field of AI that enables computers to understand, interpret, and generate human language."],
    "dataset in machine learning": ["A dataset is a collection of data used to train or test a machine learning model."],
    "model training in machine learning": ["Model training involves feeding data to a machine learning algorithm to learn patterns and make predictions."],
    "overfitting in machine learning": ["Overfitting occurs when a model learns the training data too well, performing poorly on unseen data."],










    # Encouragement
    r"i feel nervous": ["Take a deep breath. You’ve got this!"],
    r"i feel stressed": ["Try to take a break and relax. I'm here for you."],
    r"i feel lonely": ["You’re not alone; I’m here to chat with you."],

    # Closing topics
    r"thank you": ["You're welcome!", "Glad I could help!"],
    r"you're awesome": ["Thank you! You're awesome too!"],
    r"this was fun": ["I'm glad you enjoyed it!"],
}

# Function to generate responses
def get_response(user_input):
    for pattern, responses in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return random.choice(responses)
    return "I'm sorry, I don't understand that."

# Chat loop
def chat():
    print("ChatBot: Hi! I'm the ChatBot. How can I assist you? (Type 'exit' to end the chat)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("ChatBot: Goodbye!")
            break
        response = get_response(user_input)
        print("ChatBot:", response)

# Start the chat
if __name__ == "__main__":
    chat()
