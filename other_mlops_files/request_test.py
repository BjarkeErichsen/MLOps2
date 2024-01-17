import requests
response = requests.get('https://api.github.com/this-api-should-not-exist')
print(response.status_code)

response = requests.get('https://api.github.com')
print(response.status_code)


if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')

"""
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
print(response.content, type(response.content))
"""
#response message is of type "bytes"

print(response.json())

#using the GET method. Now we can provide it an argument params.
response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)
print("GET response", response)

#Writes the content of the response to a png file
#print("GET reponse png", response.content)
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
print("GET reponse png", response)
with open(r'other_mlops_files/img.png','wb') as f:
    f.write(response.content)

pload = {'username':'Olivia','password':'123'} #=payload = pload
response = requests.post('https://httpbin.org/post', data = pload)
print(response)

"""
Using the CMD to send requests. 
Sometimes its easier from CMD sometimes its easier from a script
curl -X GET "https://api.github.com"
curl -X GET -I "https://api.github.com" # if you want the status code

curl -X GET "https://imgs.xkcd.com/comics/making_progress.png" -o other_mlops_files/img.png
"""
