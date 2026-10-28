import random
from typing import Dict, Tuple, Any, List
from corridor import Corridor, Action

# =======================
# 1) Interface d'agent
# =======================

class BaseAgent:
    """Interface minimale : implémente select_action(env, obs)."""
    def __init__(self, name: str = "BaseAgent", seed: int | None = None):
        self.name = name
        if seed is not None:
            random.seed(seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Agent aléatoire : choisit uniformément une action légale."""
    def __init__(self, name: str = "RandomAgent", seed: int | None = None):
        super().__init__(name=name, seed=seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        if not actions:
            # Devrait être impossible, mais on sécurise
            raise RuntimeError("Aucune action légale disponible.")
        return random.choice(actions)


# Optionnel : exemple d'agent très simple basé sur une heuristique
class GreedyPathAgent(BaseAgent):
    """
    Heuristique: privilégie les déplacements qui rapprochent le pion de sa ligne but.
    Ne place des murs que très rarement (ou jamais).
    """
    def __init__(self, name: str = "GreedyPathAgent", wall_prob: float = 0.0, seed: int | None = None):
        super().__init__(name=name, seed=seed)
        self.wall_prob = wall_prob

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        # Filtrer les déplacements
        move_actions = [(a, dst) for (a, dst) in actions if a == "M"]
        if not move_actions:
            # Si aucun déplacement légal, choisir un mur légal (si présent)
            return random.choice(actions)

        me = 1 if obs["to_play"] == 1 else 2
        target_row = env.N - 1 if me == 1 else 0

        # Choisir le move qui minimise la distance (en ligne) vers la ligne cible
        def score_move(dst: Tuple[int, int]) -> int:
            r, c = dst
            return abs(target_row - r)

        best = min(move_actions, key=lambda x: score_move(x[1]))
        # Optionnel: parfois poser un mur
        if self.wall_prob > 0 and random.random() < self.wall_prob:
            wall_actions = [(a, w) for (a, w) in actions if a == "W"]
            if wall_actions:
                return random.choice([("W", w) for (_, w) in wall_actions])

        return ("M", best[1])


# Agents maison
class HeuristicWallAgent(BaseAgent):
    # heuristique simple : avance et met un mur si ca gene bien l'autre

    def __init__(
        self,
        name: str = "HeuristicWallAgent",
        seuil_milieu_partie: int = 7,
        marge_si_l_autre_devant: int = 1,
        penalite_pour_moi: float = 0.5,
        score_min_mur: float = 0.5,
        nb_murs_testes: int = 10,
        seed: int | None = None,
    ):
        super().__init__(name=name, seed=seed)
        self.seuil_milieu_partie = seuil_milieu_partie
        self.marge_si_l_autre_devant = marge_si_l_autre_devant
        self.penalite_pour_moi = penalite_pour_moi
        self.score_min_mur = score_min_mur
        self.nb_murs_testes = nb_murs_testes

    def _doit_regarder_les_murs(self, env: Corridor, obs: Dict) -> bool:
        # on regarde les murs que si ca vaut le coup

        joueur = obs["to_play"]
        adversaire = 2 if joueur == 1 else 1

        distance_joueur = env.shortest_path_length(joueur)
        distance_adversaire = env.shortest_path_length(adversaire)
        if distance_joueur is None or distance_adversaire is None:
            return False

        if env.move_count < self.seuil_milieu_partie:
            return False

        if obs["walls_left"][joueur] <= 0:
            return False

        # si l'autre est un peu devant on tente de le gener
        if distance_adversaire + self.marge_si_l_autre_devant <= distance_joueur:
            return True

        return False

    def _choisir_mur(self, env: Corridor, obs: Dict, actions_murs: List[Action]) -> Action | None:
        # on teste quelques murs, pas tous sinon ca explose

        if not actions_murs:
            return None

        joueur = obs["to_play"]
        adversaire = 2 if joueur == 1 else 1

        distance_joueur_avant = env.shortest_path_length(joueur)
        distance_adversaire_avant = env.shortest_path_length(adversaire)
        if distance_joueur_avant is None or distance_adversaire_avant is None:
            return None

        # on prend des murs assez proche de l'autre
        pos_adversaire = obs["p2"] if joueur == 1 else obs["p1"]
        adversaire_ligne, adversaire_col = pos_adversaire

        def distance_mur_a_adversaire(action: Action) -> int:
            _, (ligne, col, _ori) = action
            return abs(ligne - adversaire_ligne) + abs(col - adversaire_col)

        # tri simple + on prend les N premiers
        actions_murs = sorted(actions_murs, key=distance_mur_a_adversaire)
        candidats = actions_murs[: max(1, self.nb_murs_testes)]

        meilleur_score = -999.0
        meilleur_mur = None

        for action_mur in candidats:
            env_copie = env.clone()
            env_copie.step(action_mur)

            distance_joueur_apres = env_copie.shortest_path_length(joueur)
            distance_adversaire_apres = env_copie.shortest_path_length(adversaire)
            if distance_joueur_apres is None or distance_adversaire_apres is None:
                continue

            delta_joueur = distance_joueur_apres - distance_joueur_avant
            delta_adversaire = distance_adversaire_apres - distance_adversaire_avant

            # on veut monter la dist de l'autre sans trop se flinguer
            score = float(delta_adversaire) - self.penalite_pour_moi * float(delta_joueur)

            if score > meilleur_score:
                meilleur_score = score
                meilleur_mur = action_mur

        if meilleur_mur is not None and meilleur_score >= self.score_min_mur:
            return meilleur_mur

        return None

    def _choisir_deplacement(self, env: Corridor, obs: Dict, actions_moves: List[Action]) -> Action:
        # test des moves on prend celui qui baisse la dist

        joueur = obs["to_play"]

        def eval_move(action_move: Action) -> float:
            env_copie = env.clone()
            env_copie.step(action_move)
            distance = env_copie.shortest_path_length(joueur)
            if distance is None:
                return 999.0
            return float(distance)

        return min(actions_moves, key=eval_move)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("Aucune action légale disponible.")

        actions_moves = [action for action in actions if action[0] == "M"]
        actions_murs = [action for action in actions if action[0] == "W"]

        if actions_murs and self._doit_regarder_les_murs(env, obs):
            mur = self._choisir_mur(env, obs, actions_murs)
            if mur is not None:
                return mur

        if actions_moves:
            return self._choisir_deplacement(env, obs, actions_moves)

        
        return random.choice(actions)


class QLearningAgent(BaseAgent):
    # q-learning tabulaire on apprend moves + murs

    def __init__(
        self,
        name: str = "QLearningAgent",
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9965,
        min_epsilon: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__(name=name, seed=seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q[etat][action] = valeur
        self.Q: Dict[Any, Dict[Action, float]] = {}

    def obs_vers_etat(self, env: Corridor, obs: Dict) -> Any:
        # etat petit, sinon ca explose

        joueur = obs["to_play"]
        adversaire = 2 if joueur == 1 else 1

        pos_joueur = obs["p1"] if joueur == 1 else obs["p2"]
        pos_adversaire = obs["p2"] if joueur == 1 else obs["p1"]

        dist_joueur = env.shortest_path_length(joueur)
        dist_adversaire = env.shortest_path_length(adversaire)

        murs_joueur = obs["walls_left"][joueur]
        murs_adversaire = obs["walls_left"][adversaire]

        # on cap un peu la distance juste pour limiter les etats
        if dist_joueur is None:
            dist_joueur = 10
        if dist_adversaire is None:
            dist_adversaire = 10
        dist_joueur = min(dist_joueur, 10)
        dist_adversaire = min(dist_adversaire, 10)

        return (pos_joueur, pos_adversaire, dist_joueur, dist_adversaire, murs_joueur, murs_adversaire)

    def _assurer_Q(self, etat: Any, actions: List[Action]) -> Dict[Action, float]:
        if etat not in self.Q:
            self.Q[etat] = {}
        table = self.Q[etat]
        for action in actions:
            if action not in table:
                table[action] = 0.0
        return table

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("Aucune action légale disponible.")

        etat = self.obs_vers_etat(env, obs)
        table = self._assurer_Q(etat, actions)

        # epsilon greedy
        if random.random() < self.epsilon:
            return random.choice(actions)

        meilleure_action = max(actions, key=lambda action: table[action])
        return meilleure_action

    def mise_a_jour(
        self,
        etat: Any,
        action: Action,
        reward: float,
        prochain_etat: Any | None,
        actions_suivantes: List[Action] | None,
        fini: bool,
    ) -> None:
        # update q-learning standard

        table = self._assurer_Q(etat, [action])
        valeur_avant = table[action]

        if fini or (prochain_etat is None) or (not actions_suivantes):
            cible = reward
        else:
            table_suivante = self._assurer_Q(prochain_etat, actions_suivantes)
            max_suivant = max(table_suivante[action_suivante] for action_suivante in actions_suivantes)
            cible = reward + self.gamma * max_suivant

        table[action] = valeur_avant + self.alpha * (cible - valeur_avant)

    def diminuer_epsilon(self) -> None:
        # on baisse un peu a chaque episode
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



#  Entrainement

def entrainer_q_agent(
    env: Corridor,
    agent_q: QLearningAgent,
    adversaires: List[BaseAgent],
    nb_episodes: int = 1200,
    max_moves: int = 250,
    verbose: bool = False,
) -> None:
    # alterne si l'agent est J1/J2

    for episode in range(1, nb_episodes + 1):
        obs = env.reset()
        fini = False

        adversaire = adversaires[(episode - 1) % len(adversaires)]
        joueur_q = 1 if (episode % 2 == 1) else 2

        dernier_etat_q = None
        derniere_action_q = None

        while (not fini) and (env.move_count < max_moves):
            joueur = obs["to_play"]

            if joueur == joueur_q:
                etat = agent_q.obs_vers_etat(env, obs)
                action = agent_q.select_action(env, obs)

                dernier_etat_q = etat
                derniere_action_q = action

                dist_avant = env.shortest_path_length(joueur)
                obs_suiv, reward_base, fini, info = env.step(action)
                dist_apres = env.shortest_path_length(joueur)

                # reward 
                reward = float(reward_base)
                reward -= 0.02  # penalite temps

                if dist_avant is not None and dist_apres is not None:
                    reward += 0.05 * float(dist_avant - dist_apres)

                if fini:
                    prochain_etat = None
                    actions_suivantes = None
                else:
                    prochain_etat = agent_q.obs_vers_etat(env, obs_suiv)
                    actions_suivantes = env.legal_actions()

                agent_q.mise_a_jour(etat, action, reward, prochain_etat, actions_suivantes, fini)
                obs = obs_suiv

            else:
                action_adv = adversaire.select_action(env, obs)
                obs_suiv, reward_adv, fini, info = env.step(action_adv)

                # si l'autre gagne on punit notre dernier coup
                if fini and (dernier_etat_q is not None) and (derniere_action_q is not None):
                    gagnant = info.get("winner")
                    joueur_adverse = 2 if joueur_q == 1 else 1

                    if gagnant == joueur_adverse:
                        agent_q.mise_a_jour(
                            etat=dernier_etat_q,
                            action=derniere_action_q,
                            reward=-1.0,
                            prochain_etat=None,
                            actions_suivantes=None,
                            fini=True,
                        )

                obs = obs_suiv

        agent_q.diminuer_epsilon()

        if verbose and (episode % max(1, nb_episodes // 10) == 0):
            print(f"[train] ep {episode}/{nb_episodes} | eps={agent_q.epsilon:.3f} | Q=J{joueur_q}")

# =======================
# 2) Boucle de partie
# =======================

def play_game(env: Corridor, agent1: BaseAgent, agent2: BaseAgent, render: bool = False, max_moves: int = 500) -> dict:
    obs = env.reset()
    if render:
        env.render()

    agents = {1: agent1, 2: agent2}
    history: List[dict] = []

    for _ in range(max_moves):
        player = obs["to_play"]
        agent = agents[player]
        action = agent.select_action(env, obs)
        obs, reward, done, info = env.step(action)

        history.append({
            "player": player,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        })

        if render:
            env.render()

        if done:
            winner = info.get("winner")
            return {
                "winner": winner,
                "move_count": env.move_count,
                "history": history
            }

    # Sécurité: match nul si trop long 
    return {"winner": None, "move_count": env.move_count, "history": history}


def _eval_matchup(env: Corridor, agent_A: BaseAgent, agent_B: BaseAgent, n_games: int, render: bool = False) -> dict:
    # A vs B en alternant qui commence

    resultats = {"A_wins": 0, "B_wins": 0, "Draw": 0, "avg_moves": 0.0}
    total_moves = 0

    for partie in range(n_games):
        if partie % 2 == 0:
            agent_p1, agent_p2 = agent_A, agent_B
        else:
            agent_p1, agent_p2 = agent_B, agent_A

        out = play_game(env, agent_p1, agent_p2, render=render, max_moves=250)
        gagnant = out["winner"]
        total_moves += out["move_count"]

        if gagnant is None:
            resultats["Draw"] += 1
            continue

        gagnant_obj = agent_p1 if gagnant == 1 else agent_p2
        if gagnant_obj is agent_A:
            resultats["A_wins"] += 1
        else:
            resultats["B_wins"] += 1

    resultats["avg_moves"] = total_moves / float(n_games)
    return resultats


def evaluate(n_games: int = 50, render: bool = False):
    # ici on met un petit plateau, sinon q-learning galere trop

    board_size = 5
    walls_per_player = 10
    max_moves = 250

    env = Corridor(N=board_size, walls_per_player=walls_per_player)

    agent_q = QLearningAgent(
        name="QL-pur",
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9965,
        min_epsilon=0.05,
        seed=123,
    )

    agent_random = RandomAgent(name="Random-2", seed=321)
    agent_heur = HeuristicWallAgent(name="Heuristic-2", seed=456)

    print("Entrainement du QLearningAgent (moves + murs)...")
    entrainer_q_agent(
        env=env,
        agent_q=agent_q,
        adversaires=[agent_random, agent_heur],
        nb_episodes=1200,
        max_moves=max_moves,
        verbose=True,
    )

    # test en greedy
    agent_q.epsilon = 0.0

    print(f"\n=== Eval QL (A) vs Random (B) sur {n_games} parties ===")
    res_random = _eval_matchup(env, agent_q, agent_random, n_games, render=render)
    print(f"A_wins  : {res_random['A_wins']}")
    print(f"B_wins  : {res_random['B_wins']}")
    print(f"Draw    : {res_random['Draw']}")
    print(f"Avg len : {res_random['avg_moves']:.2f} coups")

    print(f"\n=== Eval QL (A) vs Heuristic (B) sur {n_games} parties ===")
    res_heur = _eval_matchup(env, agent_q, agent_heur, n_games, render=render)
    print(f"A_wins  : {res_heur['A_wins']}")
    print(f"B_wins  : {res_heur['B_wins']}")
    print(f"Draw    : {res_heur['Draw']}")
    print(f"Avg len : {res_heur['avg_moves']:.2f} coups")

    print(f"\n=== Eval Heuristic (A) vs Random (B) sur {n_games} parties ===")
    res_hr = _eval_matchup(env, agent_heur, agent_random, n_games, render=render)
    print(f"Heuristic wins : {res_hr['A_wins']}")
    print(f"Random wins    : {res_hr['B_wins']}")
    print(f"Draw           : {res_hr['Draw']}")
    print(f"Avg len        : {res_hr['avg_moves']:.2f} coups")


if __name__ == "__main__":
    # Lancer une partie unique avec rendu:
    # play_game(Corridor(), RandomAgent(), RandomAgent(), render=True)

    # Lancer une évaluation
    evaluate(n_games=100, render=False)
