import json
from src.predictive_models import (
    load_datasets, 
    build_frequency_model, 
    build_sensationalism_model,
    build_malicious_account_model,
    build_naive_realism_model
)

from src.articles import (
    evaluate_article
)

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    train_path = "data/train_set.csv"
    val_path = "data/val_set.csv"
    test_path = "data/test_set.csv"
    df_train, df_val, df_test = load_datasets(
        train_path, val_path, test_path
    )
    print("Datasets loaded successfully.")

    freq_cfg = cfg["models"]["frequency"]
    freq_model = build_frequency_model(df_train, **freq_cfg)
    print("Frequency Heuristic model trained.")

    sens_cfg = cfg["models"]["sensationalism"]
    sens_model = build_sensationalism_model(
        df_train,**sens_cfg)
    print("Sensationalism model trained.")

    ma_cfg = cfg["models"]["malicious_account"]
    ma_model = build_malicious_account_model(
        df_train, **ma_cfg)
    print("Malicious Account model trained.")       

    nr_cfg = cfg["models"]["naive_realism"]
    nr_model = build_naive_realism_model(
        df_train, **nr_cfg)
    print("Naive Realism model trained.")

    # Use trained models to predict scores for new articles
    print("\nAnalyzing article...")
    # article_url = "https://www.cnn.com/2025/10/20/politics/trump-no-kings-protests-vance-cia-analysis"  
    article_text = """
    Trump’s response to ‘No Kings’ marches only proved the protesters’ point
    Analysis by
    Stephen Collinson Profile
    Stephen Collinson
    Oct 19, 2025



    President Donald Trump arrives to greet Argentinian President Javier Milei outside the West Wing of the White House in Washington, DC, on October 14.
    President Donald Trump arrives to greet Argentinian President Javier Milei outside the West Wing of the White House in Washington, DC, on October 14. Stefani Reynolds/Bloomberg/Getty Images
    A president steeped in constitutional lore might have been offended by claims he’s acting as a king.

    But Donald Trump and his entourage responded with mockery to weekend “No Kings” protests by millions of Americans — embracing the narrative in a way that explains his growing hubris and belief that he has unchecked power.

    The president and Vice President JD Vance this weekend both posted AI memes to social media depicting Trump in a crown. The one Trump shared was a mocked-up vision of the president flying a fighter jet emblazoned with “KING TRUMP” appearing to dump raw sewage on protesters. In Vance’s troll on Bluesky, a site popular with liberals, prominent Democrats knelt as supplicants in a royal court before Trump, the divine and absolute ruler.

    It was a clever political trick. MAGA supporters can blast anyone who takes offense to the posts as lacking a sense of humor and being prim. The posts also trivialize protesters’ gravely serious claims that America is witnessing a wannabe autocrat. But they also subtly advance the thesis that Trump is all-powerful and immune from dissent, an attractive proposition to voters who like a strongman.

    Attendees sign a banner representing the US Constitution during a "No Kings" protest in Washington, DC, on Saturday.
    Attendees sign a banner representing the US Constitution during a "No Kings" protest in Washington, DC, on Saturday. Allison Robbert/AP
    Trump’s post is not only juvenile but also betrays striking contempt for tens of millions of Americans he ostensibly leads and for the concept of democratic free speech. He’s far from the first major political figure to express disdain for voters they don’t agree with. Presidents Barack Obama and Joe Biden, as well as 2016 Democratic nominee Hillary Clinton, all had their moments. And a sense that liberal elites disrespected heartland Americans was fundamental to Trump’s rise to power.

    Moreover, Democrats seem to have done little self-reflection over whether their aggressive push to implement their progressive values alienated more conservative Americans and contributed to their own defeat in 2024.

    But the president’s social media baiting would be easier to laugh off if it weren’t for multiplying actions that reinforce his pretensions to absolute power and the obliteration of constitutional curbs meant to ensure kings don’t again rule America. He’s implying there’s no room for those who don’t support him and showing he’s willing to force them into line.


    Whether such leadership is forcing the country inexorably closer to a political and social fracture or a pride-induced fall by Trump and his gang will become clear in the months to come. The stakes are especially high amid a government shutdown the president appears to have no desire to end.

    But as Trump’s behavior becomes more imperious and as opposition gathers — from street protests to more universities refusing to bend to his ideological will, and judges temporarily halting his orders for troops to enter US cities — tensions and divisions are rising.

    The president shows no sign of changing course. He called Saturday’s mass protests a “joke” and described them as “very small, very ineffective.” The people who took part were “whacked out,” Trump said.


    AD
    “When you look at those people, those are not representative of the people of our country,” he told reporters aboard Air Force One on Sunday.

    Trump’s latest monarchical moves
    Trump’s recent track record reinforces perceptions that his denial that he sees himself as a king is yet another of his uncountable untruths.

    His stunning commutation Friday night of the seven-year sentence of former Rep. George Santos, who pleaded guilty to fraud charges, was the president’s latest parody of the legal system that he unapologetically wields as a tool to help his friends and hurt his foes. This followed his demands that ex-FBI chief James Comey and New York Attorney General Letitia James be charged with crimes, which were followed by indictments scant weeks later.


    AD
    Santos on Sunday related to CNN’s Dana Bash how he’d been alerted to his impending release by other prison inmates who heard about it on TV. On “State of the Union,” Santos said, “I’m pretty confident that if President Trump had pardoned Jesus Christ off of the cross, he would have had critics. So that’s just the reality of our country.” But Rep. Nick LaLota, Santos’ former GOP colleague from New York, said of Santos: “He stole millions, defrauded an election, and his crimes (for which he pled guilty) warrant more than a three-month sentence.”

    Rep. George Santos is surrounded by journalists as he leaves the US Capitol after his fellow members of Congress voted to expel him from the House of Representatives on December 1, 2023.
    Rep. George Santos is surrounded by journalists as he leaves the US Capitol after his fellow members of Congress voted to expel him from the House of Representatives on December 1, 2023. Kevin Dietsch/Getty Images
    Pardoning and commutations are one of the few areas where the Constitution suggest presidents do have unaccountable power. But Trump’s treatment of Santos, which was far from his first highly political show of clemency, was an audacious statement that the US legal system’s reputation for blind justice now risks being tainted by made-to-order prosecutions and his regal whims. (Of course, Republicans will respond that Biden pardoned members of his own family, so all is fair game.)

    The president’s authoritarian writ is also being extended abroad.

    ► Defense Secretary Pete Hegseth announced Sunday that the US military conducted a seventh strike against a boat in the Caribbean that the administration claims was manned by narcotics traffickers. “These cartels are the Al Qaeda of the Western Hemisphere, using violence, murder and terrorism to impose their will, threaten our national security and poison our people,” Hegseth said.

    ► Trump’s promises to get tough with drugs traffickers who have caused great misery inside the US resonated with many voters last year. But the administration’s unilateral designation of traffickers as terrorists and assumption of authority to kill them without due process, while ignoring Congress’ authority over war powers, is arguably illegal and unconstitutional. It places the United States in a category of nations that mock the rule of law and opens the door for others to do the same. “All of these people have been blown up without us knowing their name, without any evidence of a crime,” Republican Sen. Rand Paul said on NBC’s “Meet the Press” on Sunday. “If they want all-out war where we kill anybody and everybody that is in the country of Venezuela or coming out, that has to have a declaration of war,” the Kentucky Republican said. “It’s something that is not pretty, very expensive, and I’m not in favor of declaring war on Venezuela, but Congress should vote. The president shouldn’t do this by himself.”

    ► Trump last week detailed his decision to authorize covert CIA operations in Venezuela, prompting debate over a possible regime-change strategy against the brutal rule of President Nicolás Maduro. On Sunday, Trump stoked speculation of further US military action in the Western Hemisphere, warning that if Colombian President Gustavo Petro didn’t “close up these killing fields” where drugs are produced, “the United States will close them up for him and it won’t be done nicely.”

    Curtailing drugs trafficking might be a laudable goal, but Trump’s methods threaten to bolster an imperial presidency and change America’s global reputation. His administration’s refusal to produce evidence of its claims or to try to build support for a potential military campaign is another example of contempt for the democratic process — one that is fueled by the genuflection of most Republicans in Congress.


    AD
    The saga is unfolding as Hegseth’s Defense Department has kicked out journalists who refused to sign the administration’s draconian new press regulations. The civilian leadership of Pentagon looks scared of scrutiny — a position that risks eroding public support for the military. That possibility is only reinforced by the administration’s determination to deploy troops on American soil.

    Protesters challenge the administration’s dismissal of demonstrations
    Trump aides tried to diminish and defame the “No Kings” protests that drew millions of people to more than 2,700 events in 50 states and were largely peaceful. White House press secretary Karoline Leavitt claimed the Democratic Party’s constituency was made up of “Hamas terrorists, illegal aliens, and violent criminals.” House Speaker Mike Johnson branded them “hate America” rallies made up of the “pro-Hamas wing” and “the Antifa people.”

    There is a seam of the Democratic base nurtured by left-wing protest movements that many centrist Americans regard as extreme. But impressions gathered around the country by CNN reporters also indicated that many people with more moderate views also took part in protests that felt like block parties to register concerns about a president many of them see as un-American. Demonstrators dressed as frogs, chickens and Statues of Liberty mocked the unhinged rhetoric of the administration about the protests just as effectively as the president’s vulgar social media posts mocked them.

    People gather along a waterfront park during a "No Kings" protest in Portland, Oregon, on Saturday.
    People gather along a waterfront park during a "No Kings" protest in Portland, Oregon, on Saturday. Jenny Kane/AP
    “We have millions of people turning out today,” Colleen Connell, executive director of the Illinois chapter of the American Civil Liberties Union, told CNN’s Fredricka Whitfield on Saturday. “There are clearly more of us than there are of those who want an autocracy or a dictatorship. And so the message is pretty clear that we, the people, you know, own this country and that we are going to respect the rule of law and we are going to stand up and we are going to protect our democracy with peaceful protest and peaceful dissent.”

    In time, Saturday’s protests may come to be seen not just as a reaction to Trump’s first nine months in power but as a harbinger of greater resistance to come.

    At the least, they show that while many of Trump’s supporters back tough-guy policies like expanding Immigrations and Custom Enforcement raids and extrajudicial executions of claimed drugs traffickers on the high seas, many other Americans don’t.

    And while warnings of encroaching authoritarianism may initially have seemed overblown, growing numbers of Americans seem to think that sitting on the sidelines is untenable. This follows a warning by Obama on Marc Maron’s final podcast that it wouldn’t be a huge sacrifice for more Americans to make their views known a year before midterm elections. “We’re not at the stage where you have to be like Nelson Mandela and be in a 10-by-12 jail cell for 27 years and break rocks,” the former president said.

    Organizers claimed 7 million protesters came out at the weekend. If that’s true, that means nearly 10% of the 75 million people who voted for Kamala Harris last November left their homes to join a protest.

    A person dressed as former US President George Washington displays a sign during a "No Kings" protest in Boston on Saturday.
    A person dressed as former US President George Washington displays a sign during a "No Kings" protest in Boston on Saturday. Brian Snyder/Reuters
    A protester dressed in a Statue of Liberty costume takes part in a "No Kings" protest in New York on Saturday.
    A protester dressed in a Statue of Liberty costume takes part in a "No Kings" protest in New York on Saturday. Olga Fedorova/AP
    Retired government worker Peggy Cole of Flint, Michigan, told CNN’s Veronica Stracqualursi she traveled to Washington for a protest because it was a “scary time.”

    “It seems to me, (Trump is) taking our government, our democracy, and dismantling it piece by piece, slowly, but surely, if we sit by and don’t do anything about it,” Cole said.

    Asked for comment about the protests, White House spokesperson Abigail Jackson wrote back: “Who cares?”

    That’s a response that breathes contempt for the values protesters claim to be protecting. But it also reveals a White House that believes its increasingly monarchical leader has no need to listen to the aspirations of those he regards as below him.

    In some ways, Trump is proving the protesters’ point.
    """

    article_analysis = evaluate_article(
        article_text,
        freq_model, 
        sens_model, 
        ma_model,
        nr_model
    )
    
    print(f"\nArticle Analysis Results:")
    print(f"Frequency Heuristic Level: {article_analysis['predicted_frequency_heuristic']}")
    print(f"Frequency Heuristic Score: {article_analysis['frequency_heuristic_score']:.3f}")
    print(f"Sensationalism Level: {article_analysis['predicted_sensationalism']}")
    print(f"Sensationalism Score: {article_analysis['sensationalism_score']:.3f}")
    print(f"Malicious Account Level: {article_analysis['predicted_malicious_account']}")
    print(f"Malicious Account Score: {article_analysis['malicious_account_score']:.3f}")
    print(f"Naive Realism Level: {article_analysis['predicted_naive_realism']}")
    print(f"Naive Realism Score: {article_analysis['naive_realism_score']:.3f}")


if __name__ == "__main__":
    main()